from models import AutoEncoder, CNNAutoEncoder, CNN_DCNN, CNN_DCNN_WN, CNN_DCNN_Spectral, CNN_DCNN_PWS
from recurrent_critic import CNN_DCNN_Interpol, RecurrentCritic
from loss_functions import reconstruction_loss, validation_set_acc
import torch
import random
import numpy as np
import warnings
import torchtext
import os
import matplotlib.pyplot as plt
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   pad_batch_and_add_EOS, \
                                   matrix_from_pretrained_embedding, \
                                   load_vocab, \
                                   autoencoder_info, \
                                   most_similar_words, \
                                   word_deletion, \
                                   random_masking, \
                                   plot_ae_loss

def word_mixup(config, model, targets, padded_batch, matrix_for_sampling, num_special_tokens):
    # interpolates between two embeddings (words and other_words)
    # where other_words are sampled nearest neighbours of words
    # interpolation weights are drawn from a folded beta distribution with (0.1, 0.1)

    forbidden_words = list(np.arange(num_special_tokens)) # speacial tokens not pretrained
    flat_batch = [item for sublist in padded_batch for item in sublist]
    num_samples = len(flat_batch)
    rinterpol_factors = [np.random.beta(0.1, 0.1) for x in range(num_samples)]
    rinterpol_factors = [x if x < 0.5 else 1-x for x in rinterpol_factors]

    # index sampling matrix with real words in batch and their sampled nearest neighbours
    rd_idx = np.random.choice(matrix_for_sampling.shape[-1])
    other_words = matrix_for_sampling[flat_batch, rd_idx]

    # correct interpolation of forbidden words both ways
    other_words = [torch.tensor(word) if (word in forbidden_words or other_word in forbidden_words) else other_word for word, other_word in zip(flat_batch, other_words)]
    rinterpol_factors = [0 if y == z else x for x, y, z in zip(rinterpol_factors, flat_batch, other_words)]

    words = torch.tensor(padded_batch).to(model.device)
    other_words = torch.LongTensor(other_words).to(model.device)
    other_words = torch.reshape(other_words, (words.shape))
    assert torch.all(torch.eq(torch.nonzero(words), torch.nonzero(other_words)) == True), "interpolation of forbidden words wrong"
    words_copy = words.detach().clone()
    other_words_copy = other_words.detach().clone()
    # obtain embedding of words and other_words
    words_embedded = model.encoder.embed_trainable_and_untrainable(words)
    other_words_embedded = model.encoder.embed_trainable_and_untrainable(other_words)
    words_embedded = words_embedded.flatten(0,1)
    other_words_embedded = other_words_embedded.flatten(0,1)

    interpol_batch = []
    old_target_shape = targets.shape
    targets = torch.flatten(targets, 0, 1)
    words_flat = torch.flatten(words_copy)
    other_words_flat = torch.flatten(other_words_copy)

    # interpolate embeddings of words with other words
    # interpolate corresponding labels
    idx = 0
    for word_embed, other_word_embed, weight in zip(words_embedded, other_words_embedded, rinterpol_factors):
        if weight > 0:
            interpol_batch.append(torch.lerp(word_embed, other_word_embed, weight).detach())
        else:
            interpol_batch.append(word_embed)
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

def plot_cprod(c_prod):
    epochs = len(c_prod)
    c_prod = np.array(c_prod)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_ae_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "CNN_DCNN_WN_"+ str(epochs) + "_epochs_lipschitz_constant.png"
    final_directory = os.path.join(directory, filename)
    temp = epochs
    epochs = []
    for i in range(temp):
        epochs.append(i)
    epochs = np.array(epochs)
    plt.xticks(range(temp))
    if temp >= 15:
        ax = plt.gca()
        temp = ax.xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::5]))
        for label in temp:
            label.set_visible(False)
    plt.plot(epochs, c_prod, label = 'Lipschitz constant')
    plt.xlabel('Epochs')
    plt.ylabel('L')
    plt.title("Autoencoder Lipschitz Development", fontsize = 10)
    plt.grid(True)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def plot_pws(pws):
    epochs = len(pws)
    pws = np.array(pws)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_ae_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "CNN_DCNN_PWS_"+ str(epochs) + "_pws.png"
    final_directory = os.path.join(directory, filename)
    temp = epochs
    epochs = []
    for i in range(temp):
        epochs.append(i)
    epochs = np.array(epochs)
    plt.xticks(range(temp))
    if temp >= 15:
        ax = plt.gca()
        temp = ax.xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::5]))
        for label in temp:
            label.set_visible(False)
    plt.yscale("log")
    plt.plot(epochs, pws, label = 'Pairwise Similarity Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title("Autoencoder PWS Development", fontsize = 10)
    plt.grid(True)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def load_cnn_dcnn(config, model_path):
    model = CNN_DCNN(config)
    model.to(config.device)
    model.load_state_dict(torch.load(model_path, map_location = config.device), strict = False)
    return model

def load_hybrid(config, model_path, teacher_forcing):
    model = CNNAutoEncoder(config, teacher_forcing = teacher_forcing)
    model.to(config.device)
    model.load_state_dict(torch.load(model_path, map_location = config.device), strict = False)
    return model

def train(config, 
          num_epochs = 20,
          model_name = "CNN_DCNN", # choose among: default_autoencoder (+_tf), cnn_autoencoder (+_tf), CNN_DCNN, CNN_DCNN_WN, CNN_DCNN_PWS
          regime = "normal", # choose among: "normal", "word-deletion", "masking", "word-mixup"
          teacher_forcing = "False",
          latent_mode = "dropout", 
          data_path = "corpus_v40k_ids.txt",
          vocab_path = "vocab_40k.txt", 
          logging_interval = 5, 
          saving_interval = 10_000,
          plotting_interval = 10_000,
          validation_size = 10_000,
          random_seed = 42):
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("loading data: {} and vocab: {}".format(data_path, vocab_path)) 
    data = load_data_from_file(data_path, 1010_000)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path, 40_000)
    config.vocab_size = len(revvocab)

    path = "/scratch/s4184416/test_new_models/saved_aes/epoch_10_model_cnn_autoencoder_tf_regime_word-mixup_latent_mode_dropout.pth"
    pretrained_cnn = False
    if pretrained_cnn:
        print("pretrained_cnn", path)
    pretrained_hybrid = True
    if pretrained_hybrid:
        print("pretrained_hybrid", path)

    config.pretrained_embedding = True
    config.word_embedding = 100
    config.encoder_dim = 100
    config.ae_batch_size = 64
    if latent_mode == "sparse":
        config.use_dropout = False
    if config.pretrained_embedding == True:
        assert config.word_embedding == 100, "glove embedding can only have dim 100, change config"
        glove = torchtext.vocab.GloVe(name='twitter.27B', dim=100) # 27B is uncased
        weights_matrix = matrix_from_pretrained_embedding(list(vocab.keys()), config.vocab_size, config.word_embedding, glove)
        if regime == "word-mixup":
            #matrix_for_sampling = most_similar_words(weights_matrix, top_k=20)
            matrix_for_sampling = torch.randint(0, 40_000, (weights_matrix.size(0), 10))
    else:
        weights_matrix = None

    if model_name == "default_autoencoder":
        model = AutoEncoder(config, weights_matrix, teacher_forcing)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
    elif model_name == "cnn_autoencoder":
        if pretrained_hybrid:
            model = load_hybrid(config, path, teacher_forcing = True)
        else:
            model = CNNAutoEncoder(config, weights_matrix, teacher_forcing)
            model = model.apply(CNNAutoEncoder.init_weights)
            model.to(model.device)
    elif model_name == "CNN_DCNN":
        if pretrained_cnn:
            model = load_cnn_dcnn(config, path)
        else:
            model = CNN_DCNN(config)
            model = model.apply(CNN_DCNN.init_weights)
        model.to(config.device)
        config.ae_learning_rate = 1e-5
    elif model_name == "CNN_DCNN_PWS":
        model = CNN_DCNN_PWS(config)
        model = model.apply(CNN_DCNN_PWS.init_weights)
        model.to(config.device)
        config.ae_learning_rate = 1e-5
    elif model_name == "CNN_DCNN_Spectral":
        model = CNN_DCNN_Spectral(config)
        model = model.apply(CNN_DCNN_Spectral.init_weights)
        model.to(config.device)
        config.ae_learning_rate = 1e-5
    elif model_name == "interpol":
        model = CNN_DCNN_Interpol(config)
        model = model.apply(CNN_DCNN_Interpol.init_weights)
        model.to(config.device)
        config.ae_learning_rate = 1e-5
        critic = RecurrentCritic(config)
    elif model_name == "CNN_DCNN_WN":
        if pretrained_cnn:
            model = load_cnn_dcnn(config, path)
        else:
            model = CNN_DCNN_WN(config)
            model = model.apply(CNN_DCNN_WN.init_weights)
        model.to(config.device)
        config.ae_learning_rate = 1e-5
        lam = 5e-9
        print("lam for cprod", lam)
    else:
        warnings.warn("Provided invalid model name. Loading default autoencoder...")
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)

    if model.name == "CNN_DCNN" or model.name == "CNN_DCNN_WN" or model.name == "CNN_DCNN_Spectral" or model.name == "CNN_DCNN_PWS":
        config.MAX_SENT_LEN = 29

    optimizer = torch.optim.Adam(lr = config.ae_learning_rate, 
                                 params = model.parameters(),
                                 betas = (config.ae_betas[0], config.ae_betas[1]),
                                 eps=1e-08)
                                
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if model.name == "interpol":
        optimizer_crit = torch.optim.Adam(lr = 5e-4, 
                                          params = model.parameters(),
                                          betas = (config.ae_betas[0], config.ae_betas[1]),
                                          eps=1e-08)
                                
        scaler_crit = torch.cuda.amp.GradScaler(enabled=True)


    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []} 
        
    model.train()
    if model.name == "interpol":
        critic.train()

    print("######################################################")
    print("######################################################")
    print("Starting AE training. Number of training epochs: {}".format(num_epochs))
    print("Logging interval:", logging_interval)
    print("Training regime", regime)
    assert config.latent_dim == config.block_dim, "GAN block dimension and latent dimension must be equal"
    iter_counter = 0
    re_list = []
    if model.name == "default_autoencoder" or model.name == "default_autoencoder_tf" or model.name == "cnn_autoencoder" or model.name == "cnn_autoencoder_tf":
        autoencoder_info(model, config)
    print("######################################################")
    print("######################################################")
    train_error_all_epochs = []
    val_error_all_epochs = []
    if model.name == "CNN_DCNN_WN":
        c_prod_list = []
    if model.name == "CNN_DCNN_PWS" or model.name == "interpol":
        pws_list = []
        mse_error = torch.nn.MSELoss()

    for epoch_idx in range(num_epochs):
        epoch_wise_loss = []
        epoch_wise_pws_list = []
        for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
            iter_counter += 1
            tf_prob = 1
            if regime == "word-deletion":
                tampered_batch = word_deletion(batch, 0.2)
                original_lens_batch = real_lengths(tampered_batch, config.MAX_SENT_LEN)
                original_lens_batch_untampered = real_lengths(batch, config.MAX_SENT_LEN)
                padded_batch = pad_batch(tampered_batch, config.MAX_SENT_LEN)
                targets = pad_batch_and_add_EOS(batch, config.MAX_SENT_LEN)
                weights = return_weights(original_lens_batch_untampered, config.MAX_SENT_LEN)
            elif regime == "normal":
                original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
                padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
                targets = pad_batch_and_add_EOS(batch, config.MAX_SENT_LEN)
                weights = return_weights(original_lens_batch, config.MAX_SENT_LEN)
            elif regime == "word-mixup":
                original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
                padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
                weights = return_weights(original_lens_batch, config.MAX_SENT_LEN)
                targets = pad_batch_and_add_EOS(batch, config.MAX_SENT_LEN)
                targets = torch.LongTensor(targets)
                targets = torch.nn.functional.one_hot(targets, num_classes = weights_matrix.size(0)).float()
                interpol_batch, targets = word_mixup(config, model, targets, padded_batch, matrix_for_sampling, config.num_special_tokens)
            elif regime == "masking":
                tampered_batch = random_masking(batch, 3, 0.2)
                original_lens_batch = real_lengths(tampered_batch, config.MAX_SENT_LEN)
                padded_batch = pad_batch(tampered_batch, config.MAX_SENT_LEN)
                targets = pad_batch_and_add_EOS(batch, config.MAX_SENT_LEN) #targets not masked
                weights = return_weights(original_lens_batch, config.MAX_SENT_LEN)
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
                if model_name == "CNN_DCNN_WN":
                    decoded_logits, c_prod = model(padded_batch, original_lens_batch, tf_prob, mixed_up_batch, use_mixup)
                    reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                    loss = reconstruction_error + c_prod * lam
                elif model_name == "CNN_DCNN" or model_name == "CNN_DCNN_Spectral":
                    decoded_logits = model(padded_batch, original_lens_batch, tf_prob, mixed_up_batch, use_mixup)
                    reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                    loss = reconstruction_error
                elif model_name == "CNN_DCNN_PWS":
                    decoded_logits, pws1, pws2 = model(padded_batch, original_lens_batch, tf_prob, mixed_up_batch, use_mixup)
                    pairwise_error = mse_error(pws1, pws2)
                    reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                    loss = reconstruction_error + pairwise_error * 100
                    epoch_wise_pws_list.append(pairwise_error.item() * 100)
                elif model_name == "interpol":
                    decoded_logits, interpol_factors, decoded_fake_logits = model(padded_batch, original_lens_batch, tf_prob, mixed_up_batch, use_mixup)
                    fake_score = critic(decoded_fake_logits)
                    real_score = critic(decoded_logits)
                    real_interpol = torch.zeros(real_score.size()).to(config.device) 
                    critic_loss_fake = mse_error(fake_score, interpol_factors)
                    critic_loss_real = mse_error(real_score, real_interpol)
                    critic_loss = critic_loss_real + critic_loss_fake
                    reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                    loss = reconstruction_error + torch.mean(fake_score)
                else:
                    decoded_logits = model(padded_batch, original_lens_batch, tf_prob, mixed_up_batch, use_mixup)
                    reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                    loss = reconstruction_error
                re_list.append(reconstruction_error.item())
                epoch_wise_loss.append(reconstruction_error.item())

            scaler.scale(loss).backward(retain_graph=True)
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if model_name == "interpol":
                print("critic_loss", critic_loss.item())
                print("fake_score", torch.mean(fake_score).item())
                scaler_crit.scale(critic_loss).backward()
                scaler_crit.unscale_(optimizer_crit)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=False)
                scaler_crit.step(optimizer_crit)
                scaler_crit.update()
                optimizer_crit.zero_grad()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())

            if iter_counter > 0 and iter_counter % logging_interval == 0:
                progress = ((batch_idx+1) * config.ae_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                if model.name == "CNN_DCNN_WN":
                    print("c_prod", c_prod.item())
                    c_prod_list.append(c_prod.item())
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | current lr: {:.6f}'\
                    .format(progress,epoch_idx, batch_idx+1, loss.item(), reconstruction_error.item(), optimizer.param_groups[0]['lr']))
                if model.name == "CNN_DCNN_PWS":
                    print("pairwise_error", pairwise_error.item() * 100)
                
        save_model(epoch_idx+11, model, regime , latent_mode)   
        if validation_size >= config.ae_batch_size:
            val_error, bleu_score = validation_set_acc(config, model, val, revvocab)
        train_error_all_epochs.append(sum(epoch_wise_loss) / len(epoch_wise_loss))
        if model.name == "CNN_DCNN_PWS":
            pws_list.append(sum(epoch_wise_pws_list) / len(epoch_wise_pws_list))
        val_error_all_epochs.append(val_error)
        write_ae_accs_to_file(model.name, regime, epoch_idx+1, sum(epoch_wise_loss) / len(epoch_wise_loss), val_error, bleu_score)

    if model.name == "CNN_DCNN_PWS":
        plot_ae_loss(train_error_all_epochs, val_error_all_epochs, regime + "_pws_", model.name)  
    else:
        plot_ae_loss(train_error_all_epochs, val_error_all_epochs, regime, model.name) 
    if model.name == "CNN_DCNN_WN":  
        plot_cprod(c_prod_list)
    if model.name == "CNN_DCNN_PWS":
        plot_pws(pws_list) 
    save_model(epoch_idx+11, model, regime, latent_mode) 
    return log_dict