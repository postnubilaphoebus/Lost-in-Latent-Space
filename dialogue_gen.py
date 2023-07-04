import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class EncoderRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.word_embedding = config.word_embedding
        self.dropout = config.keep_prob
        self.num_layers = config.num_layers
        
        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.word_embedding)
        self.gru = nn.GRU(
            input_size=self.word_embedding, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first = True, dropout=self.dropout)

    def forward(self, encoder_inputs):

        embedded = self.embedding(encoder_inputs)
        output, hidden = self.gru(embedded)

        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.word_embedding = config.word_embedding
        self.dropout = config.keep_prob
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        
        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size, embedding_dim = self.word_embedding)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.gru_cell = nn.GRUCell(
            input_size=self.hidden_size, hidden_size = self.hidden_size)
        self.gru_cell_2 = nn.GRUCell(
            input_size=self.hidden_size, hidden_size = self.hidden_size)

        self.linear_1 = nn.Linear(self.batch_size * 2, self.batch_size)
        self.linear_2 = nn.Linear(self.hidden_size, self.vocab_size)
        self.linear_reduce_dim = nn.Linear(self.batch_size * 2, self.batch_size)
        self.linear_reduce_dim_2 = nn.Linear(self.batch_size * 2, self.batch_size)
        self.projection = nn.Linear(self.vocab_size, self.hidden_size)
        self.vT = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, requires_grad=True))
        self.W1 = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.softmax = nn.Softmax(dim = -1)

    def attention(self, encoder_output, query):
        # Using attention from https://proceedings.neurips.cc/paper/2015/hash/277281aada22045c03945dcb2ca6f2ec-Abstract.html
        # Grammar as a foreign language
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()

        batch_size, time_steps, hidden_size = encoder_output.size()
        _, query_time, _ = query.size()

        if query_time < time_steps:
            query = query.repeat(1, int(time_steps / query_time), 1)

        w1 = self.W1(encoder_output)
        w2 = self.W2(query)
        result = torch.tanh(w1 + w2).flatten(0, 1).transpose(0, 1)
        u = torch.matmul(self.vT, result).transpose(0, 1).reshape(batch_size, time_steps, hidden_size)
        attn_weights = self.softmax(u)
        attns = torch.sum(attn_weights * encoder_output, dim = 1)

        return attns

    def forward(self, encoder_output, encoder_state, decoder_input, generate, loop_function = None, mc_search = True, mc_position = 0):

        attns = self.attention(encoder_output, encoder_output)

        decoder_input = decoder_input.transpose(0,1)

        sampled_words = []
        sampled_words_prob = []
        outputs = []
        dec_in_index = decoder_input
        prev_output = None
        prev = None
        state = None

        for i, inp in enumerate(decoder_input):

            word_index = dec_in_index[i]
            word_prob = prev_output

            if generate == True and prev is not None:
                loop_function = self._argmax_or_mcsearch(mc_search)

                if i > mc_position:
                    np, word_index, word_prob = loop_function(prev, i)
                else:
                    np, word_index, word_prob = inp, word_index, prev

            if i > 0:
                sampled_words.append(word_index)
                sampled_words_prob.append(word_prob)

            # merge inputs and previous attentions into vector

            inp = inp.unsqueeze(-1).repeat(1, attns.size(-1))
            inputs = torch.stack([inp, attns], dim = 0).flatten(0,1)
            inputs = self.linear_reduce_dim_2(inputs.transpose(0,1)).transpose(0,1)

            state = self.gru_cell(inputs, state)
            state = self.gru_cell_2(inputs, state)

            state = state.unsqueeze(1)
            attns = self.attention(encoder_output, state)
            state = state.squeeze(1)

            output = self.linear_2(torch.add(state, attns))
            prev_output = output
            outputs.append(output)

            if generate == True:
                prev = output

        # sample the word at the last position
        if loop_function is not None and prev is not None:
            np, word_index, word_prob = loop_function(prev, i)
            sampled_words.append(word_index)
            sampled_words_prob.append(word_prob)
        else:
            sampled_words.append(sampled_words[-1])
            sampled_words_prob.append(sampled_words_prob[-1])

        return outputs, state, [sampled_words, sampled_words_prob]

class Seq2SeqAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderRNN(config)
        self.decoder = DecoderRNN(config)
        self.config = config
        self.buckets = config.buckets

    def initialize_weights(self):
        if isinstance(self, nn.Embedding):
            sqrt3 = math.sqrt(3)
            nn.init.normal_(self.weight.data, std=sqrt3)
        else:
            pass

    def forward(self, encoder_inputs, decoder_inputs, generate, mc_search = False, mc_position = 0):
        # decoder_inputs: only "GO" symbol (first input) used when generate == True.
        # That is, we generate using next = embedding_lookup(embedding, argmax(previous_output))
        
        output, state = self.encoder.forward(encoder_inputs)
    
        outputs, state, word_and_word_prob = self.decoder.forward(output, state, decoder_inputs, generate, mc_search, mc_position)

        return outputs, state, word_and_word_prob