import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import sys
from models import AutoEncoder, CNNAutoEncoder, Generator, Critic, VariationalAutoEncoder, CNN_DCNN, CNN_DCNN_WN
from distribution_fitting import distribution_fitting, distribution_constraint
import time
import warnings
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   save_gan, \
                                   cutoff_scores, \
                                   sample_multivariate_gaussian, \
                                   plot_gan_acc, \
                                   plot_gan_loss, \
                                   sample_batch

def vae_encoding(vae, padded_batch, original_lens_batch):
    output = vae.encoder(padded_batch, original_lens_batch)
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
        if "tf" in model_file:
            model = AutoEncoder(config, weights_matrix, teacher_forcing=True)
        else:
            model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
        print("loading", model_5)
    elif model_name == "cnn_autoencoder":
        if "tf" in model_file:
            model = CNNAutoEncoder(config, weights_matrix, teacher_forcing=True)
        else:
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
            model.load_state_dict(torch.load(model_5_path, map_location=lambda storage, loc: storage))
        else:
            sys.exit("AE model path does not exist")
    else:
        sys.exit("AE path does not exist")

    return model

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

def train_gan(config,
              model_name = "default_autoencoder",
              model_file = "epoch_11_model_default_autoencoder_regime_normal_latent_mode_dropout.pth",
              num_sents = 1010_000,
              validation_size = 10_000,
              unroll_steps = 0,
              gdf = False,
              gdf_scaling_factor = 1.0,
              num_epochs = 30,
              gp_lambda = 10,
              print_interval = 10,
              n_times_critic = 10,
              data_path = "corpus_v40k_ids.txt"):
    
    config.vocab_size = 40_000
    if model_name == "variational_autoencoder":
        config.encoder_dim = 600
        config.word_embedding = 100
    else:
        config.encoder_dim = 100
        config.word_embedding = 100

    print("model_name", model_name)
    print("model_file", model_file)
    norm_type = "default"
    print("norm type", norm_type)

    if gdf == True and unroll_steps == 0:
        train_mode = "gdf"
    elif unroll_steps > 0 and gdf == False:
        train_mode = "unroll_" + str(unroll_steps)
    elif unroll_steps > 0 and gdf == True:
        train_mode = "gdf_plus_unroll_" + str(unroll_steps)
    else:
        train_mode = "normal"

    if norm_type == "snm":
        train_mode += "snm"

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

    if gdf == True:
        fitted_distribution = distribution_fitting(config, autoencoder, all_data)
        fitted_distribution = fitted_distribution.to(config.device)
        print("gdf_scaling_factor", gdf_scaling_factor)

    config.gan_batch_size = 4096
    config.n_layers = 70
    plot_cutoff = 40
    crit_activation_function = "relu"
    gen_activation_function = "relu"
    config.c_learning_rate = 1e-4
    config.g_learning_rate = 1e-4
    config.gan_betas[0] = 0
    config.gan_betas[1] = 0.9  
    print("batch size {}, block_dim {}".format(config.gan_batch_size, config.block_dim))
    print("nlayers critic {}, nlayers generator {}".format(config.n_layers, config.n_layers))
    print("n_times_critic", n_times_critic)
    print("activation G {}, activation C {}".format(gen_activation_function, crit_activation_function))
    print("unroll steps", unroll_steps)
    print("G lr", config.g_learning_rate)
    print("D lr", config.c_learning_rate)
    print("Adam betas {}, {}".format(config.gan_betas[0], config.gan_betas[1]))

    gen = Generator(config.n_layers, config.block_dim, gen_activation_function).to(config.device)
    gen = gen.apply(Generator.init_weights)
    crit = Critic(config.n_layers, config.block_dim, crit_activation_function, norm_type = norm_type).to(config.device)
    crit = crit.apply(Critic.init_weights)

    gen.train()
    crit.train()

    gen_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                 params = gen.parameters(),
                                 betas = (config.gan_betas[0], config.gan_betas[1]),
                                 eps=1e-08)
    
    crit_optim = torch.optim.Adam(lr = config.c_learning_rate, 
                                  params = crit.parameters(),
                                  betas = (config.gan_betas[0], config.gan_betas[1]),
                                  eps=1e-08) 
    
    if torch.cuda.is_available():
        print("GPU name", torch.cuda.get_device_name(0))
    else:
        import platform
        print("CPU name", platform.processor())

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
    # time_window = 200
    # moving_average_gen = [average_weights(gen).cpu().detach()] * time_window
    # moving_average_crit = [average_weights(crit).cpu().detach()] * time_window

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.gan_batch_size, all_data)):
            t0 = time.time()
            original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
            padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
            padded_batch = torch.LongTensor(padded_batch).to(config.device)
            crit_optim.zero_grad()
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
            #noise = sample_bernoulli(config)
            #noise = sample_uniform(config)
            z_fake = gen(noise)
            real_score = crit(z_real)
            fake_score = crit(z_fake.detach())
            # current_crit_weights = average_weights(crit)
            # crit_hist_penalty = MA_penalty(current_crit_weights, moving_average_crit) 
            # moving_average_crit[agnostic_idx%time_window] = current_crit_weights.cpu().detach()

            grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake, gp_lambda)
            #bc_penalty = compute_bc_penalty_mix(config, crit, z_real, z_fake)
            #bc_penalty = compute_bc_penalty(real_score)
            # unsmooth_penalty_crit = smooth_layers(crit)
            # if agnostic_idx > 0 and agnostic_idx % print_interval == 0:
            #     print("unsmooth_penalty_crit", unsmooth_penalty_crit.item())
            c_loss = - torch.mean(real_score) + torch.mean(fake_score) 
            c_loss_interval.append(c_loss.item())
            c_loss += grad_penalty
            c_loss.backward()
            crit_optim.step()

            if agnostic_idx % n_times_critic == 0:
                if unroll_steps > 0:
                    backup_crit = Critic(config.n_layers, config.block_dim, crit_activation_function).to(config.device)
                    backup_crit.load_state_dict(crit.state_dict())
                    #backup_ma = moving_average_crit
                    for i in range(unroll_steps):
                        batch = sample_batch(config.gan_batch_size, all_data)
                        original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
                        padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
                        padded_batch = torch.LongTensor(padded_batch).to(config.device)
                        crit_optim.zero_grad()
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
                        real_score_ = crit(z_real)
                        noise = sample_multivariate_gaussian(config)
                        #noise = sample_bernoulli(config)
                        #noise = sample_uniform(config)
                        with torch.no_grad():
                            z_fake = gen(noise)
                        fake_score_ = crit(z_fake)
                        grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake, gp_lambda)
                        # unsmooth_penalty_crit = smooth_layers(crit)
                        #bc_penalty = compute_bc_penalty_mix(config, crit, z_real, z_fake)
                        #bc_penalty = compute_bc_penalty(real_score)
                        # current_crit_weights = average_weights(crit)
                        # crit_hist_penalty = MA_penalty(current_crit_weights, moving_average_crit)
                        # moving_average_crit[agnostic_idx%time_window] = current_crit_weights.cpu().detach()
                        c_loss = - torch.mean(real_score_) + torch.mean(fake_score_) + grad_penalty #
                        c_loss.backward()
                        crit_optim.step()
                    noise = sample_multivariate_gaussian(config)
                    #noise = sample_bernoulli(config)
                    gen_optim.zero_grad()
                    fake_score = crit(gen(noise))
                    # unsmooth_penalty_gen = smooth_layers(gen)
                    # current_gen_weights = average_weights(gen)
                    # gen_hist_penalty = MA_penalty(current_gen_weights, moving_average_gen)
                    # moving_average_gen[agnostic_idx%time_window] = current_gen_weights.cpu().detach()
                    if gdf == True:
                        gdf_loss = distribution_constraint(fitted_distribution, gen(noise), gdf_scaling_factor)
                        g_loss = - torch.mean(fake_score) + gdf_loss #+ unsmooth_penalty_gen 
                        if agnostic_idx % print_interval == 0:
                            print("gdf_loss", gdf_loss.item())
                    else:
                        g_loss = - torch.mean(fake_score) #+ unsmooth_penalty_gen 
                    g_loss.backward()
                    gen_optim.step()
                    g_loss_interval.append(g_loss.item())
                    crit.load(backup_crit)
                    del backup_crit
                    #moving_average_crit = backup_ma
                else:
                    gen_optim.zero_grad()
                    # current_gen_weights = average_weights(gen)
                    # gen_hist_penalty = MA_penalty(current_gen_weights, moving_average_gen)
                    # moving_average_gen[agnostic_idx%time_window] = current_gen_weights.cpu().detach()
                    fake_score = crit(gen(noise))
                    #unsmooth_penalty_gen = smooth_layers(gen)
                    if gdf == True:
                        gdf_loss = distribution_constraint(fitted_distribution, gen(noise), gdf_scaling_factor)
                        g_loss = - torch.mean(fake_score) + gdf_loss 
                        if agnostic_idx % print_interval == 0:
                            print("gdf_loss", gdf_loss.item())
                    else:
                        g_loss = - torch.mean(fake_score)
                    g_loss.backward()
                    gen_optim.step()
                    g_loss_interval.append(g_loss.item())

            t3 = time.time()

            if agnostic_idx > 0 and agnostic_idx % n_times_critic == 0:
                acc_real = torch.mean(real_score)
                acc_fake = torch.mean(fake_score)
                c_loss_per_batch.append(cutoff_scores(c_loss.item(), plot_cutoff))
                g_loss_per_batch.append(cutoff_scores(g_loss.item(), plot_cutoff))
                acc_real_batch.append(cutoff_scores(acc_real.item(), plot_cutoff))
                acc_fake_batch.append(cutoff_scores(acc_fake.item(), plot_cutoff))

            if agnostic_idx > 0 and agnostic_idx % print_interval == 0:
                average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
                average_c_loss = sum(c_loss_interval) / len(c_loss_interval)
                c_loss_interval = []
                g_loss_interval = []
                progress = ((batch_idx+1) * config.gan_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                progress = round(progress, 4)
                print("Progress {}% | Generator loss {:.6f}| Critic loss {:.6f}| Acc real {:.6f} | Acc fake {:.6f} over last {} batches"
                      .format(progress, average_g_loss, average_c_loss, acc_real.item(), acc_fake.item(), print_interval))
            agnostic_idx += 1
            total_time += t3 - t0
            autoencoder_time += t2 - t1

        print("saving GAN...")
        save_gan(epoch_idx+1, autoencoder.name, gen, crit, train_mode, model_file, n_times_critic)

    print("autoencoder as fraction of time", autoencoder_time / total_time)
    plot_gan_acc(acc_real_batch, 
                 acc_fake_batch, 
                 config.gan_batch_size, 
                 config.gan_betas[0], 
                 config.c_learning_rate, 
                 autoencoder.name, 
                 model_file, 
                 n_times_critic, 
                 train_mode)
    plot_gan_loss(c_loss_per_batch, 
                  g_loss_per_batch, 
                  config.gan_batch_size, 
                  config.gan_betas[0], 
                  config.c_learning_rate, 
                  autoencoder.name, 
                  model_file, 
                  n_times_critic, 
                  train_mode)