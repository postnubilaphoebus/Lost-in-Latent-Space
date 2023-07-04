import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import copy
import pandas as pd
import utils.config as config
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
                                   read_daily_dial, \
                                   sample_batch_dial

def sample_batch_dial(batch_size, data):
    # sampling batch from data with replacement
    # PARAMS: batch_size, data
    # RETURNS: shuffled queries, shuffled replies
    random.shuffle(data)
    sample_num = random.randint(0, len(data) - batch_size - 1)
    data_batch = data[sample_num:sample_num+batch_size]
    sub_lens = [len(x) for x in data_batch]
    query_indices = [random.randint(0, x-2) for x in sub_lens]
    sampled_queries = [x[query_indices[i]] for i, x in enumerate(data_batch)]
    sampled_replies = [x[query_indices[i]+1] for i, x in enumerate(data_batch)]
    return sampled_queries, sampled_replies

def plot_mle_loss(g_loss):
    epochs = len(g_loss)
    g_loss = np.array(g_loss)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_gan_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "gan_mle_pretrain_daily.png"
    final_directory = os.path.join(directory, filename)
    temp = epochs
    epochs = []
    for i in range(temp):
        epochs.append(i)
    epochs = np.array(epochs)
    plt.plot(epochs, g_loss, label = 'generator loss')
    plt.xlabel('Batches (in 20s)')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Generator MSE loss plotted over ' + str(temp) + ' batches', fontsize = 10)
    plt.grid(True)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def save_mle_gen(generator):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_gan')
    if not os.path.exists(directory):
        os.makedirs(directory)
    generator_filename = "gen_mle_daily.pth"
    generator_directory = os.path.join(directory, generator_filename)
    torch.save(generator.state_dict(), generator_directory)

def mle_pretraining(autoencoder, gen, gen_optim, batch_size, data, num_batches):
    mse_loss = torch.nn.MSELoss()
    val_loss = []
    val_dial, dialogues = data[:100], data[100:]
    for i in range(num_batches):
        sampled_queries, sampled_replies = sample_batch_dial(batch_size, dialogues)
        sampled_queries = pad_batch(sampled_queries, config.MAX_SENT_LEN)
        sampled_queries = torch.LongTensor(sampled_queries).to(config.device)
        sampled_replies = pad_batch(sampled_replies, config.MAX_SENT_LEN)
        sampled_replies = torch.LongTensor(sampled_replies).to(config.device)
        with torch.no_grad():
            encoded_queries = autoencoder.encoder(sampled_queries)
            encoded_queries = encoded_queries.squeeze(-1)
            encoded_replies = autoencoder.encoder(sampled_replies)
            encoded_replies = encoded_replies.squeeze(-1)
        generated_replies = gen(encoded_queries)
        gen_optim.zero_grad()
        g_loss = mse_loss(generated_replies, encoded_replies)
        g_loss.backward()
        gen_optim.step()

        if i % 20 == 0 and i > 10:
            sampled_queries, sampled_replies = sample_batch_dial(batch_size, val_dial)
            sampled_queries = pad_batch(sampled_queries, config.MAX_SENT_LEN)
            sampled_queries = torch.LongTensor(sampled_queries).to(config.device)
            sampled_replies = pad_batch(sampled_replies, config.MAX_SENT_LEN)
            sampled_replies = torch.LongTensor(sampled_replies).to(config.device)

            with torch.no_grad():
                encoded_queries = autoencoder.encoder(sampled_queries)
                encoded_queries = encoded_queries.squeeze(-1)
                encoded_replies = autoencoder.encoder(sampled_replies)
                encoded_replies = encoded_replies.squeeze(-1)
                z_fake = gen(encoded_queries)
                g_loss = mse_loss(z_fake, encoded_replies)
                print("gloss val {}, i {}".format(g_loss.item(), i))
                val_loss.append(g_loss.item())

    plot_mle_loss(val_loss)
    save_mle_gen(gen)
    return gen

def load_autoencoder():
    autoencoder = CNN_DCNN(config)
    model_path = "epoch_59_model_CNN_DCNN_regime_daily.pth"
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_aes')
    model_5_path = os.path.join(saved_models_dir, model_path)
    autoencoder.load_state_dict(torch.load(model_5_path, map_location=torch.device(config.device)), strict = False)
    autoencoder.to(autoencoder.device)
    return autoencoder

def load_vocab_dial(vocab_path):
    count = 0
    vocab = {}
    vocab_file = open(vocab_path, 'r')
    while True:
        line = vocab_file.readline()
        line = line.rstrip()
        if not line:
            break
        vocab[line] = count
        count += 1
    revvocab = {v: k for k, v in vocab.items()}
    return vocab, revvocab

# File containing unsuccessful dialogue training on the dailydialogue dataset
                                   
dialogues = read_daily_dial()
val_dial, dialogues = dialogues[:100], dialogues[100:]
vocab, revvocab = load_vocab_dial("vocab_daily_dial.txt")
config.vocab_size = len(revvocab)
batch_size = 64
num_batches = 500
print("num_batches", num_batches)
config.n_layers = 12
crit_activation_function = "relu"
gen_activation_function = "relu"
print_interval = 5
n_times_critic = 5
gen = Generator(12, config.block_dim, gen_activation_function, snm = False).to(config.device)
crit = Critic(12, config.block_dim, crit_activation_function, snm = True).to(config.device)
gen = gen.apply(Generator.init_weights)
crit = crit.apply(Critic.init_weights)

config.gan_betas[0] = 0.9
config.gan_betas[1] = 0.999
config.g_learning_rate = 1e-2

gen_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                  params = gen.parameters(),
                                  betas = (config.gan_betas[0], config.gan_betas[1]),
                                  eps=1e-08, 
                                  weight_decay=1e-5)

crit_optim = torch.optim.Adam(lr = config.c_learning_rate, 
                                 params = crit.parameters(),
                                 betas = (config.gan_betas[0], config.gan_betas[1]),
                                 eps=1e-08) 

config.MAX_SENT_LEN = 29
mse_loss = torch.nn.MSELoss()
autoencoder = load_autoencoder()
autoencoder.eval()

gen = mle_pretraining(autoencoder, gen, gen_optim, batch_size, dialogues, num_batches)

sampled_queries, sampled_replies = sample_batch_dial(batch_size, dialogues)
sampled_queries = pad_batch(sampled_queries, config.MAX_SENT_LEN)
sampled_queries = torch.LongTensor(sampled_queries).to(config.device)
sampled_replies = pad_batch(sampled_replies, config.MAX_SENT_LEN)
sampled_replies = torch.LongTensor(sampled_replies).to(config.device)
with torch.no_grad():
    encoded_queries = autoencoder.encoder(sampled_queries)
    encoded_queries = encoded_queries.squeeze(-1)
    encoded_replies = autoencoder.encoder(sampled_replies)
    encoded_replies = encoded_replies.squeeze(-1)

fake_queries = gen(encoded_queries)
decoded = autoencoder.decoder(fake_queries.unsqueeze(-1))
logits = autoencoder.compute_logits(autoencoder.encoder.embedding_layer.weight.data, decoded)
fake_words = torch.argmax(logits, dim = -1)

import pdb; pdb.set_trace()

c_loss_interval = []
g_loss_interval= []
c_loss_per_batch = []
g_loss_per_batch = []
acc_real_batch = []
acc_fake_batch = []


for idx in range(num_batches):
    sampled_queries, sampled_replies = sample_batch_dial(batch_size, dialogues)
    sampled_queries = pad_batch(sampled_queries, config.MAX_SENT_LEN)
    sampled_queries = torch.LongTensor(sampled_queries).to(config.device)
    sampled_replies = pad_batch(sampled_replies, config.MAX_SENT_LEN)
    sampled_replies = torch.LongTensor(sampled_replies).to(config.device)
    #crit_optim.zero_grad()

    with torch.no_grad():
        encoded_queries = autoencoder.encoder(sampled_queries)
        encoded_queries = encoded_queries.squeeze(-1)
        encoded_replies = autoencoder.encoder(sampled_replies)
        encoded_replies = encoded_replies.squeeze(-1)

    z_fake = gen(encoded_queries)
    # real_score = crit(encoded_replies)
    # fake_score = crit(z_fake.detach())
    # c_loss = - torch.mean(real_score) + torch.mean(fake_score)# + gp_lambda * grad_penalty
    # c_loss_interval.append(c_loss.item())
    # c_loss.backward()
    # crit_optim.step()

    #if idx % n_times_critic == 0:
    gen_optim.zero_grad()
    g_loss = mse_loss(z_fake, encoded_replies)
    # fake_score = crit(gen(encoded_queries))
    # g_loss = - torch.mean(fake_score)
    g_loss.backward()
    gen_optim.step()
    g_loss_interval.append(g_loss.item())
    # acc_real = torch.mean(real_score)
    # acc_fake = torch.mean(fake_score)
    # c_loss_per_batch.append(cutoff_scores(c_loss.item(), 20))
    g_loss_per_batch.append(cutoff_scores(g_loss.item(), 20))
    # acc_real_batch.append(cutoff_scores(acc_real.item(), 20))
    # acc_fake_batch.append(cutoff_scores(acc_fake.item(), 20))

    if idx % 20 == 0:
        sampled_queries, sampled_replies = sample_batch_dial(batch_size, val_dial)
        sampled_queries = pad_batch(sampled_queries, config.MAX_SENT_LEN)
        sampled_queries = torch.LongTensor(sampled_queries).to(config.device)
        sampled_replies = pad_batch(sampled_replies, config.MAX_SENT_LEN)
        sampled_replies = torch.LongTensor(sampled_replies).to(config.device)
        #crit_optim.zero_grad()

        with torch.no_grad():
            encoded_queries = autoencoder.encoder(sampled_queries)
            encoded_queries = encoded_queries.squeeze(-1)
            encoded_replies = autoencoder.encoder(sampled_replies)
            encoded_replies = encoded_replies.squeeze(-1)
            z_fake = gen(encoded_queries)
            g_loss = mse_loss(z_fake, encoded_replies)
        
    if idx % print_interval == 0 and idx > 30:
        average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
        #average_c_loss = sum(c_loss_interval) / len(c_loss_interval)
        #c_loss_interval = []
        g_loss_interval = []
        progress = ((idx+1) / num_batches) * 100
        progress = round(progress, 4)
        print("Progress {}% | Generator loss {:.6f}|".format(progress, average_g_loss))
        # print("Progress {}% | Generator loss {:.6f}| Critic loss {:.6f}| Acc real {} | Acc fake {} over last {} batches"
        #         .format(progress, average_g_loss, average_c_loss, acc_real.item(), acc_fake.item(), print_interval))








