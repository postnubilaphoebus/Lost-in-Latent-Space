from models import CNN_DCNN, CNNAutoEncoder
from loss_functions import reconstruction_loss, validation_set_acc
import torch
import random
import numpy as np
import utils.config as config
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   pad_batch_and_add_EOS, \
                                   load_vocab, \
                                   autoencoder_info, \
                                   plot_ae_loss

def plot_ae_loss_daily(train_error, val_error, regime, ae_name):
    epochs = len(train_error)
    train_error = np.array(train_error)
    val_error = np.array(val_error)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_ae_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = ae_name + "_" + regime + "_" + str(epochs) + "_epochs_performance_daily.png"
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
    plt.plot(epochs, train_error, label = 'Train loss')
    plt.plot(epochs, val_error, label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Autoencoder Loss", fontsize = 10)
    plt.grid(True)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def save_model_daily(epoch, model, regime):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_aes')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'epoch_' + str(epoch) + '_model_' + model.name + '_regime_' + regime + '.pth'
    final_directory = os.path.join(directory, filename)
    torch.save(model.state_dict(), final_directory)

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

def train(config, 
          num_epochs = 120,
          model_name = "cnn_autoencoder", # choose among: default_autoencoder, cnn_autoencoder, CNN_DCNN, CNN_DCNN_WN
          regime = "daily", # choose among: "normal", "word-deletion", "masking", "word-mixup"
          latent_mode = "daily", 
          data_path = "corpus_daily.txt",
          vocab_path = "vocab_daily_dial.txt", 
          logging_interval = 100, 
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
    vocab, revvocab = load_vocab_dial(vocab_path)
    config.vocab_size = len(revvocab)

    config.ae_batch_size = 64
    model = CNNAutoEncoder(config, None)
    model = model.apply(CNNAutoEncoder.init_weights)
    model.to(model.device)
    if model.name == "CNN_DCNN" or model.name == "CNN_DCNN_WN":
        config.MAX_SENT_LEN = 29

    print("model loaded", model.name)
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
    print("Starting AE training. Number of training epochs: {}".format(num_epochs))
    print("Logging interval:", logging_interval)
    print("Training regime", regime)
    assert config.latent_dim == config.block_dim, "GAN block dimension and latent dimension must be equal"
    iter_counter = 0
    re_list = []

    if model.name != "CNN_DCNN" and model.name != "CNN_DCNN_WN":
        autoencoder_info(model, config)
    print("######################################################")
    print("######################################################")
    train_error_all_epochs = []
    val_error_all_epochs = []

    for epoch_idx in range(num_epochs):
        epoch_wise_loss = []
        for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
            iter_counter += 1
            tf_prob = 1
            original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
            padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
            targets = pad_batch_and_add_EOS(batch, config.MAX_SENT_LEN)
            weights = return_weights(original_lens_batch, config.MAX_SENT_LEN)
            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)
            targets = torch.LongTensor(targets).to(model.device)

            with torch.cuda.amp.autocast():
                decoded_logits = model(padded_batch, original_lens_batch, tf_prob, None, False)
                reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                loss = reconstruction_error
                re_list.append(reconstruction_error.item())
                epoch_wise_loss.append(reconstruction_error.item())

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
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | current lr: {:.6f}'\
                    .format(progress,epoch_idx, batch_idx+1, loss.item(), reconstruction_error.item(), optimizer.param_groups[0]['lr']))
                
        if epoch_idx+1 >= 5 and (epoch_idx+1) % 5 == 0:  
            save_model_daily(epoch_idx, model, regime)   
        if validation_size >= config.ae_batch_size:
            val_error, bleu_score = validation_set_acc(config, model, val, revvocab)
        train_error_all_epochs.append(sum(epoch_wise_loss) / len(epoch_wise_loss))
        val_error_all_epochs.append(val_error)

    plot_ae_loss_daily(train_error_all_epochs, val_error_all_epochs, regime, model.name)  
    save_model_daily(epoch_idx, model, regime) 
    return log_dict

if __name__ == "__main__":
    log_dict = train(config)