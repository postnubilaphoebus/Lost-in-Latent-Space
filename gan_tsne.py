from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import os
from models import Generator, CNN_DCNN, AutoEncoder, CNNAutoEncoder
import torch
import sys
import utils.config as config
import umap
import plotly.express as px

from utils.helper_functions import load_data_and_create_vocab, \
                                   prepare_data, \
                                   yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   read_daily_dial, \
                                   sample_multivariate_gaussian

file = "gen_70_layers_base_ae_unroll.pth"

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

def load_gan(config, filename):
    print("Loading pretrained generator...")
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_gan')
    model_15_path = os.path.join(saved_models_dir, filename)
    model = Generator(n_layers = 70, block_dim = 100)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_15_path):
            model.load_state_dict(torch.load(model_15_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("GAN model path does not exist")
    else:
        sys.exit("GAN path does not exist")

    return model

def load_ae(model_name, config):
    if model_name == "CNN_DCNN":
        model = CNN_DCNN(config)
        model.to(model.device)
        model_5 = "epoch_210_model_CNN_DCNN_WN_regime_normal_latent_mode_dropout.pth"
    elif model_name == "cnn_autoencoder":
        model = CNNAutoEncoder(config)
        model = model.apply(CNNAutoEncoder.init_weights)
        model.to(model.device)
        model_5 = "epoch_5_model_cnn_autoencoder_regime_normal_latent_mode_dropout.pth"
    elif model_name == "default_autoencoder":
        model = AutoEncoder(config)
        model.to(model.device)
        model_5 = "epoch_11_model_default_autoencoder_regime_normal_latent_mode_dropout.pth"
    else:
        pass
    print("Loading pretrained ae of type {}".format(model_name))
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

model_name = "default_autoencoder"
autoencoder = load_ae(model_name, config)
generator = load_gan(config, file)
if autoencoder.name == "CNN_DCNN" or autoencoder.name == "CNN_DCNN_WN":
    config.MAX_SENT_LEN = 29

data = load_data_from_file("corpus_v40k_ids.txt", 20_000)
config.gan_batch_size = 4096
data = data[-config.gan_batch_size:]
padded_batch = pad_batch(data, config.MAX_SENT_LEN)
padded_batch = torch.LongTensor(padded_batch).to(autoencoder.device)
original_lens_batch = real_lengths(padded_batch, config.MAX_SENT_LEN)

if autoencoder.name == "CNN_DCNN" or autoencoder.name == "CNN_DCNN_WN":
    with torch.no_grad():
        z_real = autoencoder.encoder(padded_batch)
        z_real = z_real.squeeze(-1)
else:
    with torch.no_grad():
        z_real, _ = autoencoder.encoder(padded_batch, original_lens_batch)

noise = sample_multivariate_gaussian(config)
with torch.no_grad():
    z_fake = generator(noise)

label_true = [1] * config.gan_batch_size
label_fake = [0] * config.gan_batch_size
labels = label_true + label_fake
z_real = z_real.cpu().detach().numpy().tolist()
z_fake = z_fake.cpu().detach().numpy().tolist()
z_all = z_real + z_fake

z_all_embedded = TSNE(n_components=2,random_state=42).fit_transform(z_all)
plt.figure(figsize=(5, 5))
for label, sample in zip(labels, z_all_embedded):
    if label == 0:
        col = "r"
    else:
        col = "b"
    plt.scatter(sample[0], sample[1], c = col, s = (5,))

plt.legend()
plt.show()

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3)
embedding = reducer.fit_transform(z_all)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'b']
for label in np.unique(labels):
    indices = np.where(labels == label)
    ax.scatter(embedding[indices,0], embedding[indices,1], embedding[indices,2], c=colors[label])
plt.show()
