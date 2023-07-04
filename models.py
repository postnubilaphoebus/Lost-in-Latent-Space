import torch
from torch import nn
import sys
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F

MAX_SENT_LEN = 28

# Notation:
# B = Batch size
# S = Sequence length
# H = Hidden dimension
# E = Embedding dimension
# V = Vocab size
# L = Latent dimension

##########################################################################################################
###                                                                                                    ###
########################################## General Model functions #######################################                                      
###                                                                                                    ###
##########################################################################################################

def create_emb_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

##########################################################################################################
###                                                                                                    ###
########################################## Autoencoder models ############################################                                         
###                                                                                                    ###
##########################################################################################################

class DefaultDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.latent_dim = config.latent_dim
        self.layer_norm = config.layer_norm
        self.hidden_dim = config.decoder_dim
        self.fc = torch.nn.Linear(self.latent_dim, self.decoder_dim)
        self.layer_normalisation = torch.nn.LayerNorm(self.decoder_dim)
        self.decoder_rnn = torch.nn.GRU(self.decoder_dim, self.decoder_dim, batch_first=True)

    def forward(self, z, true_inp, tf_prob):
        # [B, L] -> [B, H]
        output = self.fc(z)

        # [B, H] -> [B, S, H]
        output, _ = self.decoder_rnn(output.unsqueeze(1).repeat(1, MAX_SENT_LEN, 1), output.unsqueeze(0))

        if self.layer_norm == True:
            output = self.layer_normalisation(output)

        # [B, S, H] -> [S, B, H]
        output = torch.transpose(output, 1, 0)
        
        return output
    
class TeacherForcingDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.latent_dim = config.latent_dim
        self.layer_norm = config.layer_norm
        self.hidden_dim = config.decoder_dim
        self.fc = torch.nn.Linear(self.latent_dim, self.decoder_dim)
        self.layer_normalisation = torch.nn.LayerNorm(self.decoder_dim)
        self.decoder_cell = torch.nn.GRUCell(self.decoder_dim, self.decoder_dim)

    def forward(self, z, true_inp, tf_prob):
        # [B, L] -> [B, H]
        z = self.fc(z)

        # [B, H] -> [S, B, H]
        # tf during training
        if self.training:
            output = []
            true_inp = true_inp.repeat(1, 1, (self.decoder_dim // true_inp.size(-1))).transpose(1,0)
            first_inp = torch.zeros(1, true_inp.size(1),true_inp.size(2)).to(self.device)
            true_inp = torch.cat((first_inp, true_inp), 0)
            for i in range(MAX_SENT_LEN):
                hx = self.decoder_cell(z, true_inp[i])
                output.append(hx)
        # no tf during inference
        else:
            output = []
            hx = torch.zeros(z.size(0), z.size(1)).to(self.device)
            for i in range(MAX_SENT_LEN):
                hx = self.decoder_cell(z, hx)
                output.append(hx)
        output = torch.stack((output))

        if self.layer_norm == True:
            output = self.layer_normalisation(output)
        
        return output
    
class ExperimentalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_dim = config.decoder_dim
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size
        self.num_mlps = 9
        self.attn_heads = 3
        self.attn_softmax = nn.Softmax(dim = -1)
        self.mlp_layers = nn.ModuleList([nn.Linear(self.latent_dim, self.decoder_dim) \
                            for _ in range(self.num_mlps)])
        self.activation_function = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_layer = nn.Conv1d(in_channels = self.decoder_dim, out_channels = self.decoder_dim, 
                                    kernel_size = 3, groups = 1)
        self.attn_layers = nn.ModuleList([nn.Linear(self.decoder_dim, self.decoder_dim) \
                                          for _ in range(self.attn_heads)])
        self.attn_out = nn.Linear(self.decoder_dim * self.attn_heads, self.decoder_dim)
        self.upsample_layers = nn.ModuleList([nn.Linear(self.decoder_dim * 2, self.decoder_dim) \
                                              for _ in range(4)])
        self.fc = nn.Linear(self.decoder_dim, self.decoder_dim)
        
    def attention(self, x):
        layer_out = []
        for layer in self.attn_layers:
            x_out = layer(x)
            u_t = torch.tanh(x_out)
            attn_weights = self.attn_softmax(u_t)
            attended = attn_weights * x
            attended = torch.sum(attended, dim = 1)
            layer_out.append(attended)
        layer_out = torch.cat((layer_out), dim = -1)
        attended = self.attn_out(layer_out)
        attended = attended.unsqueeze(1)
        attended = attended.repeat(1, attn_weights.size(1), 1)
        return attended

    def forward(self, z, true_inp, tf_prob):
        output = []
        for layer in self.mlp_layers:
            out = self.activation_function(layer(z))
            output.append(out)
        output = torch.stack((output))
        output = torch.transpose(output, -1, 0)
        output = torch.transpose(output, 1, 0)
        output = self.conv_layer(output)
        output = torch.transpose(output, 2, 1)
        attended_out = self.attention(output)
        output = torch.cat((output, attended_out), dim = -1)
        upsampled = []
        for layer in self.upsample_layers:
            out = self.activation_function(layer(output))
            upsampled.append(out)
        upsampled = torch.cat((upsampled), dim = 1)
        output = self.fc(upsampled)
        output = torch.transpose(output, 1, 0)

        return output
    
class CNN_Encoder(nn.Module):

    def __init__(self, config, weights_matrix = None):
        super().__init__()
        assert type(config.kernel_sizes) is list, "kernel sizes must be list"
        assert len(config.kernel_sizes) == 2, "two kernel sizes required"
        self.max_pool_kernel = config.max_pool_kernel
        self.cnn_embed = config.word_embedding
        self.kernel_sizes = config.kernel_sizes 
        self.out_channels = config.out_channels
        self.latent_dim = config.latent_dim
        self.dropout_prob = config.dropout_prob
        self.use_dropout = config.use_dropout
        self.vocab_size = config.vocab_size
        self.embedding_layer = nn.Embedding(self.vocab_size, self.cnn_embed) if weights_matrix == None \
                               else create_emb_layer(weights_matrix, non_trainable=True)
        self.special_tokens = config.num_special_tokens
        self.trainable_embedding = torch.nn.Embedding(self.special_tokens, self.cnn_embed) 
        self.first_convolution_a = nn.Conv1d(in_channels = self.cnn_embed, out_channels = self.out_channels, 
                                             kernel_size = self.kernel_sizes[0], groups = 1)
        self.first_convolution_b = nn.Conv1d(in_channels = self.cnn_embed, out_channels = self.out_channels, 
                                             kernel_size = self.kernel_sizes[1], groups = 1)
        self.groupNorm_a = torch.nn.GroupNorm(10, self.out_channels)
        self.groupNorm_b = torch.nn.GroupNorm(10, self.out_channels)
        self.l_out_a = (MAX_SENT_LEN - (self.kernel_sizes[0] - 1) - (self.max_pool_kernel - 1) - 1) / self.max_pool_kernel + 1
        self.l_out_b = (MAX_SENT_LEN - (self.kernel_sizes[1] - 1) - (self.max_pool_kernel - 1) - 1) / self.max_pool_kernel + 1
        self.max_pool = nn.MaxPool1d(kernel_size = self.max_pool_kernel, stride = self.max_pool_kernel)
        self.to_latent = torch.nn.Linear(round((self.l_out_a + self.l_out_b) * self.out_channels), self.latent_dim)
        self.second_dropout = nn.Dropout(p = self.dropout_prob, inplace=False)

    def embed_trainable_and_untrainable(self, x):
        # create Boolean mask for
        # indices lower than config.num_special_tokens
        mask = x < self.special_tokens
        pretrained_batch = x.clone()
        pretrained_batch[mask] = 0
        # pass batch through frozen embedding
        embedded_batch = self.embedding_layer(pretrained_batch)
        x[~mask] = 0
        # pass batch through trainable embedding
        non_pret = self.trainable_embedding(x)
        embedded_batch[mask] = non_pret[mask]
        return embedded_batch

    def forward(self, x, mixed_up_batch = None, use_mixup = False):

        if use_mixup == False:
            embedded = self.embed_trainable_and_untrainable(x)
        else:
            embedded = mixed_up_batch

        embedded_t = torch.transpose(embedded, 2, 1)
        c_1_a = self.first_convolution_a(embedded_t)
        c_1_a = self.max_pool(c_1_a)
        c_1_a = self.groupNorm_a(c_1_a)
        c_1_a = torch.flatten(c_1_a, start_dim = 1)
        c_1_b = self.first_convolution_b(embedded_t)
        c_1_b = self.max_pool(c_1_b)
        c_1_b = self.groupNorm_b(c_1_b)
        c_1_b = torch.flatten(c_1_b, start_dim = 1)
        c_1_a = torch.transpose(c_1_a, 1, 0)
        c_1_b = torch.transpose(c_1_b, 1, 0)
        c_1 = torch.cat((c_1_a, c_1_b))
        c_1 = torch.transpose(c_1, 1, 0)

        z = self.to_latent(c_1)
        if self.use_dropout == True:
            z = self.second_dropout(z)

        return z, embedded
    
class DefaultEncoder(nn.Module):
    def __init__(self, config, weights_matrix = None):
        super().__init__()
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.layer_norm = config.layer_norm
        self.embedding_dimension = config.word_embedding
        self.encoder_dim = config.encoder_dim
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout_prob
        self.use_dropout = config.use_dropout
        self.hidden_dim = config.encoder_dim
        self.embedding_layer = nn.Embedding(config.vocab_size, self.embedding_dimension) if weights_matrix == None \
                               else create_emb_layer(weights_matrix, non_trainable=True)
        self.special_tokens = config.num_special_tokens
        self.trainable_embedding = torch.nn.Embedding(self.special_tokens, self.embedding_dimension) 
        self.bidirectional = config.bidirectional
        self.attn_multiplier = 2 if self.bidirectional else 1
        self.encoder_rnn = torch.nn.GRU(self.embedding_dimension, self.encoder_dim, batch_first=True, bidirectional = self.bidirectional)
        self.layer_normalisation = torch.nn.LayerNorm(self.encoder_dim * self.attn_multiplier)
        self.attn_softmax = nn.Softmax(dim = -1)
        self.attn_bool = config.attn_bool
        self.attn_heads = config.num_attn_heads
        self.attn_layers = nn.ModuleList([nn.Linear(self.attn_multiplier * self.encoder_dim, self.attn_multiplier * self.encoder_dim) \
                            for _ in range(self.attn_heads)])
        self.attn_out = nn.Linear(self.attn_heads * self.attn_multiplier * self.encoder_dim , self.attn_multiplier * self.encoder_dim)
        self.layer_multiplier = 4 if self.attn_bool else self.attn_multiplier # because bidirectional & attention
        self.fc_1 = torch.nn.Linear(self.encoder_dim * self.layer_multiplier, self.latent_dim) 
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def mask_padding(self, x_lens, hidden_dim):
        max_seq_len = MAX_SENT_LEN
        mask = []
        for seq_len in x_lens:
            a = [0 for _ in range(seq_len)]
            if max_seq_len - seq_len > 0:
                b = [1 for _ in range(max_seq_len - seq_len)]
                a = a + b 
            mask.append(a)
        mask = torch.BoolTensor(mask).to(self.device)
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1, 1, hidden_dim)
        return mask
    
    def embed_trainable_and_untrainable(self, x):
        # create Boolean mask for
        # indices lower than config.num_special_tokens
        mask = x < self.special_tokens
        pretrained_batch = x.clone()
        pretrained_batch[mask] = 0
        # pass batch through frozen embedding
        embedded_batch = self.embedding_layer(pretrained_batch)
        x[~mask] = 0
        # pass batch through trainable embedding
        non_pret = self.trainable_embedding(x)
        embedded_batch[mask] = non_pret[mask]
        return embedded_batch

    def attention(self, x, mask):
        # attention from Yang, 2016
        # https://aclanthology.org/N16-1174.pdf
        # using multi-head
        layer_out = []
        for layer in self.attn_layers:
            x_out = layer(x)
            u_t = torch.tanh(x_out)
            attn_weights = self.attn_softmax(u_t)
            attn_weights = attn_weights.masked_fill(mask, -1e12)
            attended = attn_weights * x
            attended = torch.sum(attended, dim = 1)
            layer_out.append(attended)
        layer_out = torch.cat((layer_out), dim = -1)
        attended = self.attn_out(layer_out)
        attended = attended.unsqueeze(1)
        attended = attended.repeat(1, attn_weights.size(1), 1)
        return attended

    def forward(self, x, x_lens, mixed_up_batch = None, use_mixup = False):

        # [B, S] -> [B, S, E]
        if use_mixup == False:
            embedded = self.embed_trainable_and_untrainable(x)
        else:
            embedded = mixed_up_batch
        # packing for efficiency and masking
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
        # [B, S, E] -> [B, S, H]
        output, _ = self.encoder_rnn(packed) 

        # unpack to extract last non-padded element
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = MAX_SENT_LEN)

        if self.layer_norm == True:
            output = self.layer_normalisation(output)

        if self.attn_bool:
            mask = self.mask_padding(x_lens, output.size(-1))
            attn_vector = self.attention(output, mask)
            output = torch.cat((output, attn_vector), dim = -1)
 
        # get the last (time-wise) hidden state of the encoder
        # [B, S, H] -> [B, H]
        context = []
        for sequence, unpadded_len in zip(output, x_lens):
            context.append(sequence[unpadded_len-1, :])
        z = torch.stack((context))
        
        # [B, H] -> [B, L]
        z = self.fc_1(z)

        if self.use_dropout == True:
            # dropout
            z = self.dropout_layer(z)
        
        return z, embedded

class AutoEncoder(nn.Module):
    def __init__(self, config, weights_matrix = None, teacher_forcing = False):
        super().__init__()
        self.name = "default_autoencoder"
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.vocab_size = config.vocab_size
        self.encoder = DefaultEncoder(config, weights_matrix)
        if teacher_forcing == False:
            self.decoder = DefaultDecoder(config)
        else:
            self.decoder = TeacherForcingDecoder(config)
            self.name += "_tf"
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)       
        else:
            pass
        
    def forward(self, x, x_lens, tf_prob = 0, mixed_up_batch = None, use_mixup = False):
        
        # [B, S] -> [B, H]
        z, embedded = self.encoder(x, x_lens, mixed_up_batch, use_mixup)
        
        # [B, L] -> [S, B, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        # [S, B, H] -> [S, B, V]
        logits = self.hidden_to_vocab(decoded)

        return logits
    
class AutoEncoder_Triad_Loss(nn.Module):
    def __init__(self, config, weights_matrix = None):
        super().__init__()
        self.name = "triad_loss_ae"
        self.device = config.device
        self.use_dropout = config.use_dropout
        self.embedding_dimension = config.word_embedding
        self.decoder_dim = 100
        self.vocab_size = config.vocab_size
        self.encoder_dim = config.encoder_dim
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout_prob
        self.attn_softmax = nn.Softmax(dim = -1)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.special_tokens = config.num_special_tokens
        self.layer_normalisation = torch.nn.LayerNorm(self.encoder_dim)
        self.attn_heads = 3
        self.encoder_rnn = torch.nn.GRU(self.embedding_dimension, self.encoder_dim, batch_first=True)
        self.attn_layers = nn.ModuleList([nn.Linear(self.encoder_dim, self.encoder_dim) \
                            for _ in range(self.attn_heads)])
        self.attn_out = nn.Linear(self.attn_heads * self.encoder_dim , self.encoder_dim)
        self.fc_1 = torch.nn.Linear(self.encoder_dim, self.latent_dim) 
        self.decoder = DefaultDecoder(config)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        self.trainable_embedding = torch.nn.Embedding(self.special_tokens, self.embedding_dimension) 
        self.embedding_layer = nn.Embedding(config.vocab_size, self.embedding_dimension) if weights_matrix == None \
                               else create_emb_layer(weights_matrix, non_trainable=True)
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)       
        else:
            pass

    def embed_trainable_and_untrainable(self, x):
        # create Boolean mask for
        # indices lower than config.num_special_tokens
        mask = x < self.special_tokens
        pretrained_batch = x.clone()
        pretrained_batch[mask] = 0
        # pass batch through frozen embedding
        embedded_batch = self.embedding_layer(pretrained_batch)
        x[~mask] = 0
        # pass batch through trainable embedding
        non_pret = self.trainable_embedding(x)
        embedded_batch[mask] = non_pret[mask]
        return embedded_batch
    
    def encoder(self, embedded, x_lens = None, packing = True):
        if packing:
            # packing for efficiency and masking
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
            # [B, S, E] -> [B, S, H]
            
        output, _ = self.encoder_rnn(embedded) 

        if packing:
            # unpack to extract last non-padded element
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = MAX_SENT_LEN)

        output = self.layer_normalisation(output)

        # [B, S, H] -> [B, H]
        context = []
        for sequence, unpadded_len in zip(output, x_lens):
            context.append(sequence[unpadded_len-1, :])
        z = torch.stack((context))
        
        # [B, H] -> [B, L]
        z = self.fc_1(z)
        
        return z
        
    def forward(self, x, x_lens, tf_prob = 0, mixed_up_batch = None, use_mixup = False):

        # embedding as function
        embedded = self.embed_trainable_and_untrainable(x)

        # encoder 
        z = self.encoder(embedded, x_lens, packing = True)
        
        # [B, L] -> [S, B, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        # [S, B, H] -> [S, B, V]
        logits = self.hidden_to_vocab(decoded)

        return logits
    
class CNNAutoEncoder(nn.Module):
    def __init__(self, config, weights_matrix = None, teacher_forcing = False):
        super().__init__()
        self.name = "cnn_autoencoder"
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.vocab_size = config.vocab_size
        self.encoder = CNN_Encoder(config, weights_matrix)
        if teacher_forcing == False:
            self.decoder = DefaultDecoder(config)
        else:
            self.decoder = TeacherForcingDecoder(config)
            self.name += "_tf"
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            pass
        
    def forward(self, x, x_lens, tf_prob = 0, mixed_up_batch = None, use_mixup = False):
        
        # [B, S] -> [B, H]
        z, embedded = self.encoder(x, mixed_up_batch, use_mixup)
        
        # [B, L] -> [S, B, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        # [S, B, H] -> [S, B, V]
        logits = self.hidden_to_vocab(decoded)

        return logits
    
class ExperimentalAutoencoder(nn.Module):
    def __init__(self, config, weights_matrix = None):
        super().__init__()
        self.name = "experimental_autoencoder"
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.vocab_size = config.vocab_size
        self.encoder = CNN_Encoder(config, weights_matrix)
        self.decoder = ExperimentalDecoder(config)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            pass

    def forward(self, x, x_lens, tf_prob = 0):
        
        # [B, S] -> [B, H]
        z, embedded = self.encoder(x)
        
        # [B, L] -> [S, B, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        # [S, B, H] -> [S, B, V]
        logits = self.hidden_to_vocab(decoded)

        return logits
    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, config, weights_matrix):
        super().__init__()
        self.name = "variational_autoencoder"
        self.device = config.device
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size
        self.encoder_dim = config.encoder_dim
        self.decoder_dim = config.decoder_dim
        self.bidirectional = config.bidirectional
        self.embedding_dimension = config.word_embedding
        self.layer_norm = config.layer_norm
        self.dropout = config.dropout_prob
        self.use_dropout = config.use_dropout
        self.num_layers = 2
        self.layer_multiplier = 2 if self.bidirectional else 1
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dimension) if weights_matrix == None \
                               else create_emb_layer(weights_matrix, non_trainable=True)
        self.special_tokens = config.num_special_tokens
        self.trainable_embedding = torch.nn.Embedding(self.special_tokens, self.embedding_dimension) 
        self.encoder_rnn = torch.nn.GRU(self.embedding_dimension, 
                                        self.encoder_dim, 
                                        num_layers = self.num_layers,
                                        batch_first=True, 
                                        bidirectional = self.bidirectional)
        self.layer_norm_encoder = torch.nn.LayerNorm(self.encoder_dim * self.layer_multiplier)
        self.to_latent = torch.nn.Linear(self.encoder_dim * self.layer_multiplier, self.latent_dim)
        self.z_mean = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.z_log_var = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.to_decoder = torch.nn.Linear(self.latent_dim, self.decoder_dim)
        self.decoder_rnn = torch.nn.GRU(self.decoder_dim, 
                                        self.decoder_dim, 
                                        num_layers = self.num_layers,
                                        batch_first=True)
        self.layer_norm_decoder = torch.nn.LayerNorm(self.decoder_dim)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)

    def embed_trainable_and_untrainable(self, x):
        # create Boolean mask for
        # indices lower than config.num_special_tokens
        mask = x < self.special_tokens
        pretrained_batch = x.clone()
        pretrained_batch[mask] = 0
        # pass batch through frozen embedding
        embedded_batch = self.embedding_layer(pretrained_batch)
        x[~mask] = 0
        # pass batch through trainable embedding
        non_pret = self.trainable_embedding(x)
        embedded_batch[mask] = non_pret[mask]
        return embedded_batch

    def encoder(self, x, x_lens, mixed_up_batch = None, use_mixup = False):

        # [B, S] -> [B, S, E]
        if use_mixup == False:
            #embedded = self.embedding_layer(x)
            embedded = self.embed_trainable_and_untrainable(x)
        else:
            embedded = mixed_up_batch

        # packing for masking and speed
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
        # [B, S, E] -> [B, S, H]
        output, _ = self.encoder_rnn(embedded)

        # unpack sequence for further processing
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = max(x_lens))

        if self.layer_norm == True:
            output = self.layer_norm_encoder(output)

        output = self.to_latent(output)

        if self.use_dropout == True:
            output = self.dropout_layer(output)

        return output

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(self.device)
        z = z_mean + eps * torch.exp(z_log_var/2.) 
        return z
    
    def holistic_regularisation(self, encoder_output):
        z_mean_list = []
        z_log_var_list = []
        for time_step in encoder_output:
            z_mean_list.append(self.z_mean(time_step))
            z_log_var_list.append(self.z_log_var(time_step))
        z_mean_list = torch.transpose(torch.stack((z_mean_list)), 1, 0)
        z_log_var_list = torch.transpose(torch.stack((z_log_var_list)), 1, 0)
        return z_mean_list, z_log_var_list
    
    def decoder(self, z):
        z = self.to_decoder(z)
        out, _ = self.decoder_rnn(z.unsqueeze(1).repeat(1,MAX_SENT_LEN,1), z.unsqueeze(0).repeat(self.num_layers, 1, 1))
        if self.layer_norm == True:
            out = self.layer_norm_decoder(out)
        logits = self.hidden_to_vocab(out)
        return logits
    
    def forward(self, x, x_lens, mixed_up_batch = None, use_mixup = False):

        # [B, S] -> [B, S, H]
        output = self.encoder(x, x_lens, mixed_up_batch, use_mixup)

        # [B, S, H] -> [S, B, H]
        output = torch.transpose(output, 1, 0)

        # [S, B, H] -> [B, S, H] (order to extract last time-step)
        z_mean_list, z_log_var_list = self.holistic_regularisation(output)

        # extract h_t-1
        z_mean_context = []
        z_log_var_context = []
        for z_mean_seq, z_log_var_seq, unpadded_len in zip(z_mean_list, z_log_var_list, x_lens):
            z_mean_context.append(z_mean_seq[unpadded_len-1, :])
            z_log_var_context.append(z_log_var_seq[unpadded_len-1, :])

        z_mean_context = torch.stack((z_mean_context))
        z_log_var_context = torch.stack((z_log_var_context))

        z = self.reparameterize(z_mean_context, z_log_var_context)

        # [B, S, H] -> [S, B, V] (order for reconstruction error)
        logits = torch.transpose(self.decoder(z), 1, 0)

        # [B, S, H] -> [S, B, H] (order for kl divergence)
        z_mean_list = torch.transpose(z_mean_list, 1, 0)
        z_log_var_list = torch.transpose(z_log_var_list, 1, 0)

        return z_mean_list, z_log_var_list, logits
    
class ConvolutionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn_embed = 300
        self.vocab_size = config.vocab_size
        self.device = config.device
        self.embedding_layer = nn.Embedding(num_embeddings = self.vocab_size, 
                                            embedding_dim = self.cnn_embed, 
                                            max_norm = 1, 
                                            norm_type = 2)
        self.conv_1 = nn.Conv1d(in_channels = 300,
                                out_channels = 300,
                                kernel_size = 5,
                                stride = 2, 
                                padding = 1)
        self.conv_2 = nn.Conv1d(in_channels = 300, 
                                out_channels = 600, 
                                kernel_size = 5, 
                                stride = 2)
        self.conv_3 = nn.Conv1d(in_channels = 600, 
                                out_channels = 100, 
                                kernel_size = 5,
                                stride = 2)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(600)

    def forward(self, x):
        x = self.embedding_layer(x).transpose(2, 1)
        x = F.relu(self.bn1(self.conv_1(x)))
        x = F.relu(self.bn2(self.conv_2(x)))
        # Gaussian experiment (commented out):
        # if self.training:
        #     x = torch.tanh(self.conv_3(x) + 0.05 * torch.normal(torch.zeros(x.size(0), self.conv_3.out_channels, 1), torch.ones(x.size(0), self.conv_3.out_channels, 1)).to(self.device))
        # else:
        #     x = torch.tanh(self.conv_3(x))
        x = torch.tanh(self.conv_3(x))
        return x
    
class ConvolutionEncoder_PWS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn_embed = 300
        self.vocab_size = config.vocab_size
        self.device = config.device
        self.embedding_layer = nn.Embedding(num_embeddings = self.vocab_size, 
                                            embedding_dim = self.cnn_embed, 
                                            max_norm = 1, 
                                            norm_type = 2)
        self.conv_1 = nn.Conv1d(in_channels = 300,
                                out_channels = 300,
                                kernel_size = 5,
                                stride = 2, 
                                padding = 1)
        self.conv_2 = nn.Conv1d(in_channels = 300, 
                                out_channels = 600, 
                                kernel_size = 5, 
                                stride = 2)
        self.conv_3 = nn.Conv1d(in_channels = 600, 
                                out_channels = 100, 
                                kernel_size = 5,
                                stride = 2)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(600)

    def cos_sim(self, x, y):
        return torch.dot(x, y)

    def pairwise_cos_sim(self, embedded, final_layer):
        embedded_comparisons = []
        for i, element1 in enumerate(embedded):
            for j, element2 in enumerate(embedded):
                if j <= i:
                    continue
                cs = self.cos_sim(element1.flatten(), element2.flatten())
                # no norm div because embedding layer normed
                embedded_comparisons.append(cs)
        final_layer_comparisons = []
        for i, element1 in enumerate(final_layer):
            for j, element2 in enumerate(final_layer):
                if j <= i:
                    continue
                cs = self.cos_sim(element1, element2)
                cs /= (torch.norm(element1, 2) * torch.norm(element2, 2))
                final_layer_comparisons.append(cs)
        embedded_comparisons = torch.stack((embedded_comparisons))
        final_layer_comparisons = torch.stack((final_layer_comparisons))
        return embedded_comparisons, final_layer_comparisons

    def forward(self, x):
        x_embedded = self.embedding_layer(x).transpose(2, 1)
        x = F.relu(self.bn1(self.conv_1(x_embedded)))
        x = F.relu(self.bn2(self.conv_2(x)))
        x = torch.tanh(self.conv_3(x))
        input_pws, hidden_pws = self.pairwise_cos_sim(x_embedded, x.squeeze(-1))
        return x, input_pws, hidden_pws

class DeconvolutionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.deconv_1 = nn.ConvTranspose1d(in_channels = 100,
                                           out_channels = 600,
                                           kernel_size = 5,
                                           stride = 2)
        self.deconv_2 = nn.ConvTranspose1d(in_channels = 600,
                                           out_channels = 300,
                                           kernel_size = 5,
                                           stride = 2)
        self.deconv_3 = nn.ConvTranspose1d(in_channels = 300, 
                                           out_channels = 300, 
                                           kernel_size = 5,
                                           stride = 2)
        self.bn1 = nn.BatchNorm1d(600)
        self.bn2 = nn.BatchNorm1d(300)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.deconv_1(x)))
        x = F.relu(self.bn2(self.deconv_2(x)))
        x = self.deconv_3(x)
        x_norm_dec = torch.norm(x, 2, dim = 1, keepdim=True)
        x_normed_dec = x / x_norm_dec
        return x_normed_dec
    
class CNN_DCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_DCNN"
        self.tau = 0.01
        self.device = config.device
        self.encoder = ConvolutionEncoder(config)
        self.decoder = DeconvolutionDecoder(config)

    def init_weights(self, activation_function = 'relu'):
        if isinstance(self, nn.Conv1d) and self.out_channels != 100:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d) and self.in_channels != 300:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        else:
            pass

    def compute_logits(self, embedding_weights, x_dec):
        x_dec = x_dec.transpose(2, 1)
        embedding_weights = embedding_weights.unsqueeze(0).repeat(x_dec.size(0), 1, 1).transpose(2, 1)
        prob_logits = (x_dec @ embedding_weights) / self.tau
        prob_logits = prob_logits.transpose(1, 0)
        return prob_logits

    def forward(self, x, y, z, t, u):
        x_enc = self.encoder(x)
        x_normed_dec = self.decoder(x_enc)
        logits = self.compute_logits(self.encoder.embedding_layer.weight.data, x_normed_dec)
        return logits
    
class CNN_DCNN_PWS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_DCNN_PWS"
        self.tau = 0.01
        self.device = config.device
        self.encoder = ConvolutionEncoder_PWS(config)
        self.decoder = DeconvolutionDecoder(config)

    def init_weights(self, activation_function = 'relu'):
        if isinstance(self, nn.Conv1d) and self.out_channels != 100:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d) and self.in_channels != 300:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        else:
            pass

    def compute_logits(self, embedding_weights, x_dec):
        x_dec = x_dec.transpose(2, 1)
        embedding_weights = embedding_weights.unsqueeze(0).repeat(x_dec.size(0), 1, 1).transpose(2, 1)
        prob_logits = (x_dec @ embedding_weights) / self.tau
        prob_logits = prob_logits.transpose(1, 0)
        return prob_logits

    def forward(self, x, y, z, t, u):
        x_enc, emb_pws, hidden_pws = self.encoder(x)
        x_normed_dec = self.decoder(x_enc)
        logits = self.compute_logits(self.encoder.embedding_layer.weight.data, x_normed_dec)
        return logits, emb_pws, hidden_pws
    
class ConvolutionEncoder_WN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn_embed = 300
        self.max_lipschitz = 10
        self.vocab_size = config.vocab_size
        self.embedding_layer = nn.Embedding(num_embeddings = self.vocab_size, 
                                            embedding_dim = self.cnn_embed, 
                                            max_norm = 1, 
                                            norm_type = 2)
        self.conv_1 = nn.Conv1d(in_channels = 300,
                                out_channels = 300,
                                kernel_size = 5,
                                stride = 2, 
                                padding = 1)
        self.conv_2 = nn.Conv1d(in_channels = 300, 
                                out_channels = 600, 
                                kernel_size = 5, 
                                stride = 2)
        self.conv_3 = nn.Conv1d(in_channels = 600, 
                                out_channels = 100, 
                                kernel_size = 5,
                                stride = 2)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(600)

    def norm_layer(self, W):
        frobenius_norm = torch.sqrt(torch.sum(torch.square(torch.abs(W.weight))))
        scale = 1.0 / torch.max(torch.ones(1), frobenius_norm / self.max_lipschitz)
        return scale, frobenius_norm

    def forward(self, x):
        s1, f1a = self.norm_layer(self.conv_1)
        s2, f2a = self.norm_layer(self.conv_2)
        s3, f3a = self.norm_layer(self.conv_3)
        x = self.embedding_layer(x).transpose(2, 1)
        x = F.relu(self.bn1(self.conv_1(x)) * s1)
        x = F.relu(self.bn2(self.conv_2(x)) * s2)
        x = torch.tanh(self.conv_3(x) * s3)
        return x, [f1a, f2a, f3a]

class DeconvolutionDecoder_WN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.softplus = nn.Softplus()
        self.max_lipschitz = 10
        self.dropout = torch.nn.Dropout(0.2)
        self.deconv_1 = nn.ConvTranspose1d(in_channels = 100,
                                           out_channels = 600,
                                           kernel_size = 5,
                                           stride = 2)
        self.deconv_2 = nn.ConvTranspose1d(in_channels = 600,
                                           out_channels = 300,
                                           kernel_size = 5,
                                           stride = 2)
        self.deconv_3 = nn.ConvTranspose1d(in_channels = 300, 
                                           out_channels = 300, 
                                           kernel_size = 5,
                                           stride = 2)
        self.bn1 = nn.BatchNorm1d(600)
        self.bn2 = nn.BatchNorm1d(300)
    
    def norm_layer(self, W):
        frobenius_norm = torch.sqrt(torch.sum(torch.square(torch.abs(W.weight))))
        scale = 1.0 / torch.max(torch.ones(1), frobenius_norm / self.max_lipschitz)
        return scale, frobenius_norm
    
    def forward(self, x):
        s1, f1b = self.norm_layer(self.deconv_1)
        s2, f2b = self.norm_layer(self.deconv_2)
        _, f3b = self.norm_layer(self.deconv_3)
        x = F.relu(self.bn1(self.dropout(self.deconv_1(x)) * s1))
        x = F.relu(self.bn2(self.dropout(self.deconv_2(x)) * s2))
        x = self.deconv_3(x)
        x_norm_dec = torch.norm(x, 2, dim = 1, keepdim=True)
        x_normed_dec = x / x_norm_dec
        return x_normed_dec, [f1b, f2b, f3b]
    
class CNN_DCNN_WN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_DCNN_WN"
        self.tau = 0.01
        self.device = config.device
        self.encoder = ConvolutionEncoder_WN(config)
        self.decoder = DeconvolutionDecoder_WN(config)

    def init_weights(self, activation_function = 'relu'):
        if isinstance(self, nn.Conv1d) and self.out_channels != 100:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d) and self.in_channels != 300:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        else:
            pass

    def compute_logits(self, embedding_weights, x_dec):
        x_dec = x_dec.transpose(2, 1)
        embedding_weights = embedding_weights.unsqueeze(0).repeat(x_dec.size(0), 1, 1).transpose(2, 1)
        prob_logits = (x_dec @ embedding_weights) / self.tau
        prob_logits = prob_logits.transpose(1, 0)
        return prob_logits

    def forward(self, x, y, z, t, u):
        x_enc, c_list_enc = self.encoder(x)
        x_normed_dec, c_list_dec = self.decoder(x_enc)
        c_list_enc.extend(c_list_dec)
        c_prod = torch.prod(torch.stack((c_list_enc)))
        logits = self.compute_logits(self.encoder.embedding_layer.weight.data, x_normed_dec)
        return logits, c_prod
    
class ConvolutionEncoderSpectral(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn_embed = 300
        self.vocab_size = config.vocab_size
        self.dropout = torch.nn.Dropout(p = 0.5)
        self.embedding_layer = nn.Embedding(num_embeddings = self.vocab_size, 
                                            embedding_dim = self.cnn_embed, 
                                            max_norm = 1, 
                                            norm_type = 2)
        self.conv_1 = spectral_norm(nn.Conv1d(in_channels = 300,
                                              out_channels = 300,
                                              kernel_size = 5,
                                              stride = 2, 
                                              padding = 1))
        self.conv_2 = spectral_norm(nn.Conv1d(in_channels = 300, 
                                              out_channels = 600, 
                                              kernel_size = 5, 
                                              stride = 2))
        self.conv_3 = spectral_norm(nn.Conv1d(in_channels = 600, 
                                              out_channels = 100, 
                                              kernel_size = 5,
                                              stride = 2))

    def forward(self, x):
        x = self.embedding_layer(x).transpose(2, 1)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = torch.tanh(self.dropout(self.conv_3(x)))
        return x
    
class DeconvolutionDecoderSpectral(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.deconv_1 = spectral_norm(nn.ConvTranspose1d(in_channels = 100,
                                                         out_channels = 600,
                                                         kernel_size = 5,
                                                         stride = 2))
        self.deconv_2 = spectral_norm(nn.ConvTranspose1d(in_channels = 600,
                                                         out_channels = 300,
                                                         kernel_size = 5,
                                                         stride = 2))
        self.deconv_3 = nn.ConvTranspose1d(in_channels = 300, 
                                                         out_channels = 300, 
                                                         kernel_size = 5,
                                                         stride = 2)
    
    def forward(self, x):
        x = F.relu(self.deconv_1(x))
        x = F.relu(self.deconv_2(x))
        x = self.deconv_3(x)
        x_norm_dec = torch.norm(x, 2, dim = 1, keepdim=True)
        x_normed_dec = x / x_norm_dec
        return x_normed_dec
    
class CNN_DCNN_Spectral(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "CNN_DCNN_Spectral"
        self.tau = 0.01
        self.device = config.device
        self.encoder = ConvolutionEncoderSpectral(config)
        self.decoder = DeconvolutionDecoderSpectral(config)

    def init_weights(self, activation_function = 'relu'):
        if isinstance(self, nn.Conv1d) and self.out_channels != 100:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d) and self.in_channels != 300:
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0)
        elif isinstance(self, nn.ConvTranspose1d):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                self.bias.data.fill_(0)
        else:
            pass

    def compute_logits(self, embedding_weights, x_dec):
        x_dec = x_dec.transpose(2, 1)
        embedding_weights = embedding_weights.unsqueeze(0).repeat(x_dec.size(0), 1, 1).transpose(2, 1)
        prob_logits = (x_dec @ embedding_weights) / self.tau
        prob_logits = prob_logits.transpose(1, 0)
        return prob_logits

    def forward(self, x, y, z, t, u):
        x_enc = self.encoder(x)
        x_normed_dec = self.decoder(x_enc)
        logits = self.compute_logits(self.encoder.embedding_layer.weight.data, x_normed_dec)
        return logits
    
##########################################################################################################
###                                                                                                    ###
########################################## GAN models ############################################     ###
###                                                                                                    ###
##########################################################################################################
    
class Block(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1, norm_type = "default"):
        super().__init__()
        
        if activation_function == "relu":
            if norm_type == "snm":
                self.net = nn.Sequential(
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                    nn.ReLU(True),
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                )
            elif norm_type == "batch":
                self.net = nn.Sequential(
                    nn.Linear(block_dim, block_dim),
                    nn.BatchNorm1d(block_dim),
                    nn.ReLU(True),
                    nn.Linear(block_dim, block_dim),
                    nn.BatchNorm1d(block_dim)
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(block_dim, block_dim),
                    nn.ReLU(True),
                    nn.Linear(block_dim, block_dim),
                )
        elif activation_function == "leaky_relu":
            if norm_type == "snm":
                self.net = nn.Sequential(
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                    nn.LeakyReLU(slope, True),
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                )
            elif norm_type == "batch":
                self.net = nn.Sequential(
                    nn.Linear(block_dim, block_dim),
                    nn.BatchNorm1d(block_dim),
                    nn.LeakyReLU(slope, True),
                    nn.Linear(block_dim, block_dim),
                    nn.BatchNorm1d(block_dim)
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(block_dim, block_dim),
                    nn.LeakyReLU(slope, True),
                    nn.Linear(block_dim, block_dim),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
    
    def forward(self, x):
        return self.net(x) + x
    
class X_Skip_Block(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1):
        super().__init__()
        if activation_function == "relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
            )
        elif activation_function == "leaky_relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
    
    def forward(self, x):
        return self.net(x) + 3 * x
    
class LongerBlock(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1):
        super().__init__()
        if activation_function == "relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
            )
        elif activation_function == "leaky_relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
    
    def forward(self, x):
        return self.net(x) + x
    
class RecursiveLayerNormBlock(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1):
        super().__init__()
        if activation_function == "relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
            )
        elif activation_function == "leaky_relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
        self.inner_norm = nn.LayerNorm(block_dim)
        self.outer_norm = nn.LayerNorm(block_dim)
    
    def forward(self, x):
        return self.outer_norm(x + self.inner_norm(x + self.net(x)))

class Generator(nn.Module):
    
    def __init__(self, n_layers, block_dim, activation_function = 'relu', slope = 0.1, norm_type = "default"):
        super().__init__()
        if norm_type == "batch":
            print("warning: only use batchnorm on WGAN without gradient penalty")
        self.net = nn.Sequential(
            *[Block(block_dim, activation_function, slope, norm_type) for _ in range(n_layers)]
        )

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.orthogonal_(self.weight, gain = 0.8)
            if self.bias is not None:
                self.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    
    def __init__(self, n_layers, block_dim, activation_function = 'relu', slope = 0.1, norm_type = "default"):
        super().__init__()
        if norm_type == "batch":
            print("warning: only use batchnorm on WGAN without gradient penalty")
        self.net = nn.Sequential(
            *[Block(block_dim, activation_function, slope, norm_type) for _ in range(n_layers)]
        )

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.orthogonal_(self.weight, gain = 0.8)
            if self.bias is not None:
                self.bias.data.fill_(0.01)
        
    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()  

    def forward(self, x):
        return self.net(x)
    
class Block_WN(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1, snm = True):
        super().__init__()
        self.layer_1 = nn.Linear(block_dim, block_dim)
        self.layer_2 = nn.Linear(block_dim, block_dim)
        self.relu = nn.ReLU()

    def norm_layer(self, W):
        inf_norm = torch.max(torch.sum(torch.abs(W.weight), 1))
        one_norm = torch.max(torch.sum(torch.abs(W.weight), 0))
        scale = torch.sqrt(one_norm * inf_norm)
        return scale
    
    def forward(self, x):
        c1 = self.norm_layer(self.layer_1)
        c2 = self.norm_layer(self.layer_2)
        x = self.relu(self.layer_1(x) / c1)
        x = self.layer_2(x) / c2
        return x

class Generator_WN(nn.Module):
    def __init__(self, n_layers, block_dim, activation_function = 'relu', slope = 0.1, snm = True):
        super().__init__()
        self.net = nn.Sequential(
            *[Block_WN(block_dim, activation_function, slope, snm) for _ in range(n_layers)]
        )
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
           torch.nn.init.orthogonal_(self.weight)
           if self.bias is not None:
               self.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
    
class Critic_WN(nn.Module):
    
    def __init__(self, n_layers, block_dim, activation_function = 'relu', slope = 0.1, snm = True):
        super().__init__()
        self.net = nn.Sequential(
            *[Block_WN(block_dim, activation_function, slope, snm) for _ in range(n_layers)]
        )

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.orthogonal_(self.weight, gain = 0.8)
            if self.bias is not None:
                self.bias.data.fill_(0.01)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()  

    def forward(self, x):
        return self.net(x)