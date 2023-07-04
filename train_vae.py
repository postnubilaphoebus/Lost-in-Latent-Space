from models import VariationalAutoEncoder
from loss_functions import reconstruction_loss, validation_set_acc
import torch
import random
import numpy as np
import warnings
import torchtext
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   pad_batch_and_add_EOS, \
                                   average_over_nonpadded, \
                                   matrix_from_pretrained_embedding, \
                                   load_vocab, \
                                   most_similar_words, \
                                   word_deletion, \
                                   random_masking, \
                                   plot_ae_loss, \
                                   plot_kl_div

def word_mixup(config, model, targets, padded_batch, matrix_for_sampling, num_special_tokens):
    # interpolates between two embeddings (words and other_words)
    # where other_words are sampled nearest neighbours of words
    # interpolation weights are drawn from [0, 0.3]

    forbidden_words = list(np.arange(num_special_tokens)) # speacial tokens not pretrained
    flat_batch = [item for sublist in padded_batch for item in sublist]
    num_samples = len(flat_batch)
    rinterpol_factors = [np.random.beta(0.4, 0.4) for x in range(num_samples)]
    rinterpol_factors = [x if x < 0.5 else 1-x for x in rinterpol_factors]

    # index sampling matrix with real words in batch and their sampled nearest neighbours
    rd_idx = np.random.choice(matrix_for_sampling.shape[-1])
    other_words = matrix_for_sampling[flat_batch, rd_idx]

    # correct interpolation of forbidden words both ways
    other_words = [word if (word in forbidden_words or other_word in forbidden_words) else other_word for word, other_word in zip(flat_batch, other_words)]

    words = torch.LongTensor(padded_batch).to(model.device)
    other_words = torch.LongTensor(other_words).to(model.device)
    other_words = torch.reshape(other_words, (words.shape))
    assert torch.all(torch.eq(torch.nonzero(words), torch.nonzero(other_words)) == True), "interpolation of forbidden words wrong"

    # obtain embedding of words and other_words
    with torch.no_grad():
        words_embedded = model.embed_trainable_and_untrainable(words)
        other_words_embedded = model.embed_trainable_and_untrainable(other_words)
        words_embedded = words_embedded.flatten(0,1)
        other_words_embedded = other_words_embedded.flatten(0,1).detach()

    interpol_batch = []
    old_target_shape = targets.shape
    targets = targets.flatten(0,1)
    words_flat = words.flatten()
    other_words_flat = other_words.flatten()

    # interpolate embeddings of words with other words
    # interpolate corresponding labels
    idx = 0
    for word_embed, other_word_embed, weight in zip(words_embedded, other_words_embedded, rinterpol_factors):
        interpol_batch.append(torch.lerp(word_embed, other_word_embed, weight))
        targets[idx, words_flat[idx]] -= weight
        targets[idx, other_words_flat[idx]] = weight
        idx += 1

    interpol_batch = torch.stack((interpol_batch)).reshape(words.size(0), words.size(1), config.word_embedding)
    targets = targets.reshape(old_target_shape).cpu().detach()
    targets = targets.to(model.device)

    return interpol_batch, targets

def write_ae_accs_to_file(model_name, train_regime, epoch_num, train_error, val_error, val_bleu):
    with open("ae_results.txt", "a") as f:
        f.write("\n")
        f.write("##################################################################################################################################" + "\n")
        f.write("model: {} train_regime: {} \n".format(model_name, train_regime))
        f.write("epoch_num: {}, train error {}, val error {}, val_bleu {} \n".format(epoch_num, train_error, val_error, val_bleu))
        f.write("##################################################################################################################################" + "\n" + "\n")
    f.close()

def kl_loss(weights, z_mean_list, z_log_var_list):
    weights = torch.transpose(weights, 1, 0)
    kl_div = []
    for weight, z_mean, z_log_var in zip(weights, z_mean_list, z_log_var_list):
        kl_div_t = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
        kl_div_t = torch.mean(kl_div_t, dim = 1)
        kl_div.append(weight * kl_div_t)
    kl_div = torch.stack((kl_div))
    kl_div = torch.sum(kl_div, dim = 0) # sum over seqlen
    kl_div = average_over_nonpadded(kl_div, weights, 0) # average over seqlen
    return torch.mean(kl_div)

def train(config, 
          num_epochs = 40,
          regime = "normal", # choose among: "normal", "word-deletion", "masking", "word-mixup"
          data_path = "corpus_v40k_ids.txt",
          vocab_path = "vocab_40k.txt", 
          logging_interval = 100, 
          saving_interval = 10_000,
          plotting_interval = 10_000,
          validation_size = 10_000,
          random_seed = 42):
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    kl_div_factor = 1.0
    print("kl_div_factor", kl_div_factor)

    print("loading data: {} and vocab: {}".format(data_path, vocab_path)) 
    data = load_data_from_file(data_path, 1010_000)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path, 40_000)
    config.vocab_size = len(revvocab)

    config.pretrained_embedding = True
    config.word_embedding = 100
    config.encoder_dim = 600
    config.ae_batch_size = 128
    latent_mode = "dropout"

    if config.pretrained_embedding == True:
        assert config.word_embedding == 100, "glove embedding can only have dim 100, change config"
        glove = torchtext.vocab.GloVe(name='twitter.27B', dim=100) # 27B is uncased
        weights_matrix = matrix_from_pretrained_embedding(list(vocab.keys()), config.vocab_size, config.word_embedding, glove)
        if regime == "word-mixup":
            matrix_for_sampling = most_similar_words(weights_matrix, top_k=10)
    else:
        weights_matrix = None

    model = VariationalAutoEncoder(config, weights_matrix)
    model = model.apply(VariationalAutoEncoder.init_weights)
    model = model.to(model.device)

    optimizer = torch.optim.Adam(lr = config.ae_learning_rate, 
                                 params = model.parameters(),
                                 betas = (config.ae_betas[0], config.ae_betas[1]),
                                 eps=1e-08)
                                
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []} 
        
    model.train()

    print("######################################################")
    print("######################################################")
    print("Starting VAE training. Number of training epochs: {}".format(num_epochs))
    print("Logging interval:", logging_interval)
    assert config.latent_dim == config.block_dim, "GAN block dimension and latent dimension must be equal"
    iter_counter = 0
    re_list = []
    kl_list = []

    print("######################################################")
    print("######################################################")
    train_error_all_epochs = []
    val_error_all_epochs = []
    kl_div_all_epochs = []

    for epoch_idx in range(num_epochs):
        epoch_wise_loss = []
        epoch_wise_kl = []
        for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
            iter_counter += 1

            if regime == "word-deletion":
                tampered_batch = word_deletion(batch, 0.2)
                original_lens_batch = real_lengths(tampered_batch)
                original_lens_batch_untampered = real_lengths(batch)
                padded_batch = pad_batch(tampered_batch)
                targets = pad_batch_and_add_EOS(batch)
                weights = return_weights(original_lens_batch_untampered)
            elif regime == "normal":
                original_lens_batch = real_lengths(batch)
                padded_batch = pad_batch(batch)
                targets = pad_batch_and_add_EOS(batch)
                weights = return_weights(original_lens_batch)
            elif regime == "word-mixup":
                original_lens_batch = real_lengths(batch)
                padded_batch = pad_batch(batch)
                weights = return_weights(original_lens_batch)
                targets = pad_batch_and_add_EOS(batch)
                targets = torch.LongTensor(targets)
                targets = torch.nn.functional.one_hot(targets, num_classes = weights_matrix.size(0)).float()
                interpol_batch, targets = word_mixup(config, model, targets, padded_batch, matrix_for_sampling, config.num_special_tokens)
            elif regime == "masking":
                tampered_batch = random_masking(batch, 3, 0.2)
                original_lens_batch = real_lengths(tampered_batch)
                padded_batch = pad_batch(tampered_batch)
                targets = pad_batch_and_add_EOS(batch) #targets not masked
                weights = return_weights(original_lens_batch)
            else:
                pass

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)

            if regime != "word-mixup":
                targets = torch.LongTensor(targets).to(model.device)
                use_mixup = False
                mixed_up_batch = None
            else:
                targets = targets.to(model.device)
                use_mixup = True
                mixed_up_batch = interpol_batch

            with torch.cuda.amp.autocast():
                z_mean_list, z_log_var_list, decoded_logits = model(padded_batch, original_lens_batch, mixed_up_batch, use_mixup)
                reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                kl_div = kl_loss(weights, z_mean_list, z_log_var_list)
                loss = reconstruction_error + kl_div_factor * kl_div
                re_list.append(reconstruction_error.item())
                epoch_wise_loss.append(reconstruction_error.item())
                kl_list.append(kl_div.item())
                epoch_wise_kl.append(kl_div.item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())

            if iter_counter > 0 and iter_counter % logging_interval == 0:
                progress = ((batch_idx+1) * config.ae_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | current lr: {:.6f} | kl divergence {:.4f}'\
                      .format(progress,epoch_idx, batch_idx+1, loss.item(), reconstruction_error.item(), optimizer.param_groups[0]['lr'], kl_div.item()))
                
        save_model(epoch_idx+1, model, regime, latent_mode)
        if validation_size >= config.ae_batch_size:
            val_error, bleu_score = validation_set_acc(config, model, val, revvocab)
        write_ae_accs_to_file(model.name, regime, epoch_idx+1, sum(epoch_wise_loss) / len(epoch_wise_loss), val_error, bleu_score)
        train_error_all_epochs.append(sum(epoch_wise_loss) / len(epoch_wise_loss))
        val_error_all_epochs.append(val_error)
        if epoch_idx > 0:
            kl_div_all_epochs.append(sum(epoch_wise_kl) / len(epoch_wise_kl))

    plot_ae_loss(train_error_all_epochs, val_error_all_epochs, regime, model.name)
    plot_kl_div(regime, model.name, kl_div_all_epochs)  

    return log_dict