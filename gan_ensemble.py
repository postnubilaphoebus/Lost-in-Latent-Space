import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import copy
import pandas as pd
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
                                   write_accs_to_file

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

def compute_grad_penalty(config, critic, real_data, fake_data):
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
    return grad_penalty

def compute_bc_penalty(real_score, beta = 10_000_000, m = 0.5):
    normed_matrix_minus_m = torch.linalg.matrix_norm(real_score) - m
    max = torch.max(normed_matrix_minus_m, 0)[0]
    penalty = beta * max
    return penalty

def compute_bc_penalty_mix(config, critic, real_data, fake_data, beta = 10_000_000, m = 0.5):
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

def smooth_layers(model, factor = 1000):
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
    return torch.sum(torch.stack((frobenius_list))) * factor

def train_gan(config,
              model_name =  "default_autoencoder",
              model_file = "epoch_11_model_default_autoencoder_regime_normal_latent_mode_dropout.pth",
              num_sents = 1010_000,
              validation_size = 10_000,
              unroll_steps = 0,
              norm_data = False,
              gdf = False,
              gdf_scaling_factor = 1.0,
              num_epochs = 160,
              gp_lambda = 10,
              print_interval = 10,
              plotting_interval = 50_000, 
              n_times_critic = 10,
              data_path = "corpus_v40k_ids.txt", 
              vocab_path = "vocab_40k.txt"):
    
    print("ensemble")
    
    config.vocab_size = 40_000
    print("gp_lambda", gp_lambda)
    print("num_epochs", num_epochs)
    autoencoder = load_ae(model_name, model_file, config)
    autoencoder.eval()
    if autoencoder.name == "CNN_DCNN" or autoencoder.name == "CNN_DCNN_WN":
        config.MAX_SENT_LEN = 29
    data = load_data_from_file(data_path, num_sents)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    config.gan_batch_size = 64
    config.n_layers = 40
    if config.n_layers >= 40:
        plot_cutoff = 40
    crit_activation_function = "relu"
    gen_activation_function = "relu"
    config.c_learning_rate = 1e-4
    config.g_learning_rate = 1e-4
    if unroll_steps > 0 :
        config.g_learning_rate = 1e-4
        config.c_learning_rate = 1e-4
        config.gan_betas[0] = 0.5
        config.gan_betas[1] = 0.9
        config.n_layers = 40
    else:
        config.gan_betas[0] = 0
    print("batch size {}, block_dim {}".format(config.gan_batch_size, config.block_dim))
    print("nlayers critic {}, nlayers generator {}".format(config.n_layers, config.n_layers))
    print("n_times_critic", n_times_critic)
    print("activation G {}, activation C {}".format(gen_activation_function, crit_activation_function))
    print("unroll steps", unroll_steps)
    print("G lr", config.g_learning_rate)
    print("D lr", config.c_learning_rate)
    print("Adam betas {}, {}".format(config.gan_betas[0], config.gan_betas[1]))
    print("Using WGAN with spectral norm (WGAN-SN)")

    gen1 = Generator(config.n_layers, config.block_dim, gen_activation_function, snm = False).to(config.device)
    gen2 = Generator(config.n_layers, config.block_dim, gen_activation_function, snm = False).to(config.device)
    gen3 = Generator(config.n_layers, config.block_dim, gen_activation_function, snm = False).to(config.device)
    crit1 = Critic(config.n_layers, config.block_dim, crit_activation_function, snm = False).to(config.device)
    crit2 = Critic(config.n_layers, config.block_dim, crit_activation_function, snm = False).to(config.device)
    crit3 = Critic(config.n_layers, config.block_dim, crit_activation_function, snm = False).to(config.device)
    gen1 = gen1.apply(Generator.init_weights)
    gen2 = gen2.apply(Generator.init_weights)
    gen3 = gen3.apply(Generator.init_weights)
    crit1 = crit1.apply(Critic.init_weights)
    crit2 = crit2.apply(Critic.init_weights)
    crit3 = crit3.apply(Critic.init_weights)
    gen1.train()
    gen2.train()
    gen3.train()
    crit1.train()
    crit2.train()
    crit3.train()
    gen1_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                 params = gen1.parameters(),
                                 betas = (config.gan_betas[0], config.gan_betas[1]),
                                 eps=1e-08)
    gen2_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                 params = gen2.parameters(),
                                 betas = (config.gan_betas[0], config.gan_betas[1]),
                                 eps=1e-08)
    gen3_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                 params = gen1.parameters(),
                                 betas = (config.gan_betas[0], config.gan_betas[1]),
                                 eps=1e-08)
    crit1_optim = torch.optim.Adam(lr = config.c_learning_rate, 
                                  params = crit1.parameters(),
                                  betas = (config.gan_betas[0], config.gan_betas[1]),
                                  eps=1e-08) 
    crit2_optim = torch.optim.Adam(lr = config.c_learning_rate, 
                                  params = crit2.parameters(),
                                  betas = (config.gan_betas[0], config.gan_betas[1]),
                                  eps=1e-08) 
    crit3_optim = torch.optim.Adam(lr = config.c_learning_rate, 
                                  params = crit3.parameters(),
                                  betas = (config.gan_betas[0], config.gan_betas[1]),
                                  eps=1e-08) 
    
    c_loss_interval = []
    g_loss_interval= []
    c_loss_per_batch = []
    g_loss_per_batch = []
    acc_real_batch = []
    acc_fake_batch = []
    
    agnostic_idx = 0
    total_time = 0
    autoencoder_time = 0
    progress =  0.0
    half_dim = config.latent_dim // 2

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.gan_batch_size*3, all_data)):
            t0 = time.time()

            original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
            padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
            padded_batch = torch.LongTensor(padded_batch).to(config.device)
            crit1_optim.zero_grad()
            crit2_optim.zero_grad()
            crit3_optim.zero_grad()
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

            z_real1 = z_real[:config.gan_batch_size]
            z_real2 = z_real[config.gan_batch_size:config.gan_batch_size*2]
            z_real3 = z_real[config.gan_batch_size*2:]
            t2 = time.time()
            noise1 = sample_multivariate_gaussian(config)
            noise2 = sample_multivariate_gaussian(config)
            noise3 = sample_multivariate_gaussian(config)

            z_fake1 = gen1(noise1)
            z_fake2 = gen2(noise2)
            z_fake3 = gen2(noise3)
            crit1_gen1_score = torch.mean(crit1(z_fake1.detach()))
            crit1_gen2_score = torch.mean(crit1(z_fake2.detach()))
            crit1_gen3_score = torch.mean(crit1(z_fake3.detach()))
            crit2_gen1_score = torch.mean(crit2(z_fake1.detach()))
            crit2_gen2_score = torch.mean(crit2(z_fake2.detach()))
            crit2_gen3_score = torch.mean(crit2(z_fake3.detach()))
            crit3_gen1_score = torch.mean(crit3(z_fake1.detach()))
            crit3_gen2_score = torch.mean(crit3(z_fake2.detach()))
            crit3_gen3_score = torch.mean(crit3(z_fake3.detach()))

            crit1_real = torch.mean(crit1(z_real1))
            crit2_real = torch.mean(crit2(z_real2))
            crit3_real = torch.mean(crit3(z_real3))

            grad_penalty_crit1_gen1 = compute_grad_penalty(config, crit1, z_real1, z_fake1)
            grad_penalty_crit1_gen2 = compute_grad_penalty(config, crit1, z_real1, z_fake2)
            grad_penalty_crit1_gen3 = compute_grad_penalty(config, crit1, z_real1, z_fake3)
            grad_penalty_crit1 = (grad_penalty_crit1_gen1 + grad_penalty_crit1_gen2 + grad_penalty_crit1_gen3) / 3

            grad_penalty_crit2_gen1 = compute_grad_penalty(config, crit2, z_real2, z_fake1)
            grad_penalty_crit2_gen2 = compute_grad_penalty(config, crit2, z_real2, z_fake2)
            grad_penalty_crit2_gen3 = compute_grad_penalty(config, crit2, z_real2, z_fake3)
            grad_penalty_crit2 = (grad_penalty_crit2_gen1 + grad_penalty_crit2_gen2 + grad_penalty_crit2_gen3) / 3

            grad_penalty_crit3_gen1 = compute_grad_penalty(config, crit3, z_real3, z_fake1)
            grad_penalty_crit3_gen2 = compute_grad_penalty(config, crit3, z_real3, z_fake2)
            grad_penalty_crit3_gen3 = compute_grad_penalty(config, crit3, z_real3, z_fake3)
            grad_penalty_crit3 = (grad_penalty_crit3_gen1 + grad_penalty_crit3_gen2 + grad_penalty_crit3_gen3) / 3

            crit1_loss = (crit1_gen1_score + crit1_gen2_score + crit1_gen3_score) / 3 - crit1_real 
            crit2_loss = (crit2_gen1_score + crit2_gen2_score + crit2_gen3_score) / 3 - crit2_real 
            crit3_loss = (crit3_gen1_score + crit3_gen2_score + crit3_gen3_score) / 3 - crit3_real 

            average_crit_loss = (crit1_loss.item() + crit2_loss.item() + crit3_loss.item()) / 3
            crit1_loss += grad_penalty_crit1 * gp_lambda
            crit2_loss += grad_penalty_crit2 * gp_lambda
            crit3_loss += grad_penalty_crit3 * gp_lambda
            c_loss_interval.append(average_crit_loss)

            crit1_loss.backward(retain_graph=True)
            crit2_loss.backward(retain_graph=True)
            crit3_loss.backward()

            crit1_optim.step()
            crit2_optim.step()
            crit3_optim.step()

            if agnostic_idx % n_times_critic == 0:
                gen1_optim.zero_grad()
                gen2_optim.zero_grad()
                gen3_optim.zero_grad()
                noise1 = sample_multivariate_gaussian(config)
                noise2 = sample_multivariate_gaussian(config)
                noise3 = sample_multivariate_gaussian(config)

                gen1_score = torch.mean(crit1(gen1(noise1))) + torch.mean(crit2(gen1(noise1))) + torch.mean(crit3(gen1(noise1)))
                gen1_score /= 3

                gen2_score = torch.mean(crit1(gen2(noise2))) + torch.mean(crit2(gen2(noise2))) + torch.mean(crit3(gen2(noise2)))
                gen2_score /= 3
                
                gen3_score = torch.mean(crit1(gen3(noise3))) + torch.mean(crit2(gen3(noise3))) + torch.mean(crit3(gen3(noise3)))
                gen3_score /= 3

                gen1_score.backward(retain_graph=True)
                gen2_score.backward(retain_graph=True)
                gen3_score.backward()

                gen1_optim.step()
                gen2_optim.step()
                gen3_optim.step()

                average_gen_loss = (gen1_score.item() + gen2_score.item() + gen3_score.item()) / 3
                g_loss_interval.append(average_gen_loss)

            t3 = time.time()

            if agnostic_idx > 0 and agnostic_idx % print_interval == 0:
                average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
                average_c_loss = sum(c_loss_interval) / len(c_loss_interval)
                c_loss_interval = []
                g_loss_interval = []
                progress = ((batch_idx+1) * config.gan_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                progress = round(progress, 4)
                print("Progress {}% | Generator loss {:.6f}| Critic loss {:.6f}| over last {} batches"
                      .format(progress, average_g_loss, average_c_loss, print_interval))
            agnostic_idx += 1
            total_time += t3 - t0
            autoencoder_time += t2 - t1


    print("autoencoder as fraction of time", autoencoder_time / total_time)
