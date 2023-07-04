import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import hnswlib
import sys
from models import AutoEncoder, CNNAutoEncoder, Generator, Critic, VariationalAutoEncoder, CNN_DCNN, CNN_DCNN_WN
from distribution_fitting import distribution_fitting, distribution_constraint
import matplotlib.pyplot as plt
import time
import warnings
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   save_gan, \
                                   cutoff_scores, \
                                   find_min_and_max, \
                                   plot_singular_values, \
                                   normalise, \
                                   re_scale, \
                                   sample_multivariate_gaussian, \
                                   sample_bernoulli, \
                                   plot_gan_acc, \
                                   plot_gan_loss, \
                                   sample_batch, \
                                   singular_values, \
                                   write_accs_to_file, \
                                   sample_uniform

def vae_encoding(vae, padded_batch, original_lens_batch):
    output = vae.encoder(padded_batch, original_lens_batch) #consumes 97% of function time
    output = torch.transpose(output, 1, 0)

    z_mean_list, z_log_var_list = vae.holistic_regularisation(output)

    # extract h_t-1
    z_mean_context = []
    z_log_var_context = []
    for z_mean_seq, z_log_var_seq, unpadded_len in zip(z_mean_list, z_log_var_list, original_lens_batch):
        z_mean_context.append(z_mean_seq[unpadded_len-1, :])
        z_log_var_context.append(z_log_var_seq[unpadded_len-1, :])
    
    z_mean_context = torch.stack((z_mean_context))
    z_log_var_context = torch.stack((z_log_var_context))
    z = vae.reparameterize(z_mean_context, z_log_var_context)

    return z

def load_ae(model_name, model_file, config):
    weights_matrix = None
    model_5 = model_file
    if model_name == "default_autoencoder":
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
        print("loading", model_5)
    elif model_name == "cnn_autoencoder":
        model = CNNAutoEncoder(config, weights_matrix)
        model = model.apply(CNNAutoEncoder.init_weights)
        model.to(model.device)
        print("loading", model_5)
    elif model_name == "CNN_DCNN":
        model = CNN_DCNN(config)
        model = model.apply(CNN_DCNN.init_weights)
        model.to(model.device)
        print("loading", model_5)
    elif model_name == "CNN_DCNN_WN":
        model = CNN_DCNN_WN(config)
        model = model.apply(CNN_DCNN_WN.init_weights)
        model.to(model.device)
        print("loading", model_5)
    elif model_name == "variational_autoencoder":
        model = VariationalAutoEncoder(config, weights_matrix)
        model = model.apply(VariationalAutoEncoder.init_weights)
        model.to(model.device)
        print("loading", model_5)
    else:
        warnings.warn("Provided invalid model name. Loading default autoencoder...")
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
        print("loading", model_5)

    print("Loading pretrained ae of type {}...".format(model_name))
    print("Model file", model_file)
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_aes')
    model_5_path = os.path.join(saved_models_dir, model_5)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_5_path):
            model.load_state_dict(torch.load(model_5_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("AE model path does not exist")
    else:
        sys.exit("AE path does not exist")

    return model
    
def load_gan(config):
    print("Loading pretrained generator...")
    print("loading epoch 30 unroll")
    model_15 = 'generator_epoch_30normal_ncrit_5_CNN_DCNN_epoch_50_model_CNN_DCNN_regime_normal_latent_mode_dropout.pth_model.pth'
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_gan')
    model_15_path = os.path.join(saved_models_dir, model_15)
    model = Generator(config.n_layers, config.block_dim)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_15_path):
            model.load_state_dict(torch.load(model_15_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("GAN model path does not exist")
    else:
        sys.exit("GAN path does not exist")

    return model

def load_crit(config):
    print("Loading pretrained disc...")
    print("loading epoch 30 unroll")
    model_15 = 'critic_epoch_30unroll_10_ncrit_1_CNN_DCNN_epoch_50_model_CNN_DCNN_regime_normal_latent_mode_dropout.pth_model.pth'
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_gan')
    model_15_path = os.path.join(saved_models_dir, model_15)
    model = Critic(config.n_layers, config.block_dim)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_15_path):
            model.load_state_dict(torch.load(model_15_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("GAN model path does not exist")
    else:
        sys.exit("GAN path does not exist")

    return model

def save_gen(epoch, autoencoder_name, generator):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_gan')
    if not os.path.exists(directory):
        os.makedirs(directory)
    generator_filename = "imle.pth"
    generator_directory = os.path.join(directory, generator_filename)
    torch.save(generator.state_dict(), generator_directory)

def compute_grad_penalty(config, critic, real_data, fake_data, gp_lambda):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1))).to(config.device)
    sample = alpha * real_data + (1-alpha) * fake_data
    sample.requires_grad_(True)
    score = critic(sample)
    outputs = torch.FloatTensor(B, config.latent_dim).fill_(1.0)
    outputs.requires_grad_(False)
    outputs = outputs.to(config.device)
    grads = autograd.grad(
        outputs=score,
        inputs=sample,
        grad_outputs=outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_penalty = ((grads.norm(2, dim=1) - 1.) ** 2).mean()
    return grad_penalty * gp_lambda

def compute_bc_penalty(real_score, beta = 2, m = 0.5):
    normed_matrix_minus_m = torch.linalg.matrix_norm(real_score) - m
    max = torch.max(normed_matrix_minus_m, 0)[0]
    penalty = beta * max
    return penalty

def compute_bc_penalty_mix(config, critic, real_data, fake_data, beta = 1, m = 0.5):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1))).to(config.device)
    sample = alpha * real_data + (1-alpha) * fake_data
    sample.requires_grad_(True)
    score = critic(sample)
    normed_matrix_minus_m = torch.linalg.matrix_norm(score) - m
    max = torch.max(normed_matrix_minus_m, 0)[0]
    penalty = beta * max
    return penalty

def average_weights(model):
    sum = 0
    for p in model.parameters():
        sum += torch.sum(p)
    return sum

def MA_penalty(current_weights, weight_history, ma_lamda = 0.1):
    penalty = torch.square(current_weights - (sum(weight_history) / len(weight_history)))
    return penalty * ma_lamda

def smooth_layers(model):
    frobenius_list = []
    idx = 0
    prev_layer_weight = model.net[0].net[0].weight
    prev_layer_bias = model.net[0].net[0].bias
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            if idx == 0:
                idx = 1
                continue
            f_norm_weight = torch.square(torch.linalg.matrix_norm(layer.weight - prev_layer_weight, ord='fro')) / (2 * layer.in_features)
            f_norm_bias = torch.sum(torch.square(layer.bias - prev_layer_bias)) / (2 * layer.in_features)
            frobenius_list.append(f_norm_weight + f_norm_bias)
            prev_layer_weight = layer.weight
            prev_layer_bias = layer.bias
    return torch.sum(torch.stack((frobenius_list)))

def plot_gen_loss(loss):
    epochs = len(loss)
    c_loss = np.array(loss)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_gan_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = "imle_loss.png"
    final_directory = os.path.join(directory, file_name)
    temp = epochs
    epochs = []
    for i in range(temp):
        epochs.append(i)
    epochs = np.array(epochs)
    plt.plot(epochs, c_loss, label = 'Generator loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Generator loss plotted over ' + str(temp) + ' batches', fontsize = 10)
    plt.grid(True)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def train_imle(config,
              model_name =  "default_autoencoder",
              model_file = "epoch_11_model_default_autoencoder_regime_normal_latent_mode_dropout.pth",
              num_sents = 1010_000,
              validation_size = 10_000,
              num_epochs = 10,
              data_path = "corpus_v40k_ids.txt", 
              vocab_path = "vocab_40k.txt"):
    
    config.vocab_size = 40_000
    if model_name == "variational_autoencoder":
        config.encoder_dim = 600
        config.word_embedding = 100
    else:
        config.encoder_dim = 100
        config.word_embedding = 100

    autoencoder = load_ae(model_name, model_file, config)
    autoencoder.eval()
    if autoencoder.name == "CNN_DCNN" or autoencoder.name == "CNN_DCNN_WN":
        config.MAX_SENT_LEN = 29

    data = load_data_from_file(data_path, num_sents)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    config.gan_batch_size = 4096

    gen = Generator(70, config.block_dim, "relu", norm_type = "default").to(config.device)
    gen = gen.apply(Generator.init_weights)
    gen.train()
    config.g_learning_rate = 1e-3
    gen_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                 params = gen.parameters(),
                                 betas = (config.gan_betas[0], config.gan_betas[1]),
                                 eps=1e-08, 
                                 weight_decay = 1e-6)
    
    g_loss_interval= []
    g_loss_per_batch = []
    loss_fn = torch.nn.MSELoss()
    agnostic_idx = 0
    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.gan_batch_size, all_data)):
            t0 = time.time()
            original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
            padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
            padded_batch = torch.LongTensor(padded_batch).to(config.device)
            t1 = time.time()
            with torch.no_grad():
                if autoencoder.name == "default_autoencoder":
                    z_real, _ = autoencoder.encoder(padded_batch, original_lens_batch)
                elif autoencoder.name == "cnn_autoencoder":
                    z_real, _ = autoencoder.encoder(padded_batch)
                elif autoencoder.name == "variational_autoencoder":
                    z_real = vae_encoding(autoencoder, padded_batch, original_lens_batch)
                elif autoencoder.name == "CNN_DCNN_WN":
                    z_real, _ = autoencoder.encoder(padded_batch)
                    z_real = z_real.squeeze(-1)
                elif autoencoder.name == "CNN_DCNN":
                    z_real = autoencoder.encoder(padded_batch)
                    z_real = z_real.squeeze(-1)
                else:
                    pass
            t2 = time.time()
            noise = sample_multivariate_gaussian(config)
            z_fake = gen(noise)
            # finding the nearest _real_ neighbour for each artificially generated datapoint
            # using HSNW
            z_fake_ann = z_fake.detach().numpy()
            z_real_ann = z_real.detach().numpy()
            p = hnswlib.Index(space='l2', dim=100)
            p.init_index(max_elements=noise.size(0), ef_construction=100, M=16)
            p.set_ef(10)
            p.add_items(z_fake_ann)
            labels, distances = p.knn_query(z_real_ann, k = 1)
            labels = torch.from_numpy(labels.astype(int).flatten())
            # matching the nearest neighbours of real and fake data points
            nearest_real_neighbours = torch.index_select(z_real, 0, labels).to(config.device)
            loss = loss_fn(z_fake, nearest_real_neighbours)
            gen_optim.zero_grad()
            loss.backward()
            gen_optim.step()
            g_loss_interval.append(loss.item())
            agnostic_idx += 1
            if agnostic_idx % 5 == 0:
                average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
                g_loss_per_batch.append(cutoff_scores(average_g_loss, 0.02))
                g_loss_interval = []
                progress = ((batch_idx+1) * config.gan_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                print("loss {:.6f}".format(average_g_loss))
                print("progress {:.2f}%".format(progress*100))

    save_gen(1, autoencoder.name, gen)
    plot_gen_loss(g_loss_per_batch)
