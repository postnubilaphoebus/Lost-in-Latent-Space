import torch
import utils.config as config
from train_ae import train as train_ae
from train_gan import train_gan
from test_ae import test
from train_vae import train as train_vae
from test_gan import test as test_gan
from imle import train_imle

def main(*args, **kwargs):

    log_dict = train_ae(config, 
                        num_epochs = 5, 
                        model_name = "cnn_autoencoder", 
                        regime = "word-mixup",
                        teacher_forcing = "True")
    # train_gan(config, 
    #           model_name = "default_autoencoder",
    #           model_file = "epoch_11_model_default_autoencoder_regime_normal_latent_mode_dropout.pth",
    #           num_sents = 1010_000,
    #           num_epochs = 25,
    #           gp_lambda = 10)
    #test(config)
    #test_gan(config)
    #train_imle(config)

if __name__ == "__main__":
    main(config)