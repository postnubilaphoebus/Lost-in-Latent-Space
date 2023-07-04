from models import AutoEncoder, CNNAutoEncoder, VariationalAutoEncoder
import os
import torch
import sys
import utils.config as config
from utils.helper_functions import load_data_from_file, \
                                   yieldBatch, \
                                   real_lengths, \
                                   pad_batch, \
                                   update, \
                                   finalize, \
                                   normalise


def load_model(config, model_name, model_path, weights_matrix = None):
    if os.path.isfile(model_path):
        if model_name == "default_autoencoder":
            model = AutoEncoder(config, weights_matrix)
            model = model.apply(AutoEncoder.init_weights)
            model.to(model.device)
        elif model_name == "cnn_autoencoder":
            model = CNNAutoEncoder(config, weights_matrix)
            model = model.apply(CNNAutoEncoder.init_weights)
            model.to(model.device)
        elif model_name == "variational_autoencoder":
            model = VariationalAutoEncoder(config, weights_matrix)
            model = model.apply(VariationalAutoEncoder.init_weights)
            model.to(model.device)
        else:
            sys.exit("no valid model name provided")
        model.load_state_dict(torch.load(model_path, map_location = model.device), strict = False)
    else:
        sys.exit("ae model path does not exist")
    return model

def distribution_constraint(fitted_distribution, mini_batch, scaling_factor = 1.0):
    # Applies global distribution fitting by Gong et al. in https://arxiv.org/abs/2212.01521
    # L(G) = 1/n (||mu_g - mu_r||_1 + ||sig_g - sig_r||_1) (see p. 4)
    batch_mean = torch.mean(mini_batch, dim = 1)
    batch_sigma = torch.std(mini_batch, dim = 1)
    batch_stats = torch.stack((batch_mean, batch_sigma))
    
    cnt = 0
    constraint_sum = 0
    for real_stats, gen_stats in zip(fitted_distribution, batch_stats):
        mu_diff = torch.abs(gen_stats[0] - real_stats[0])
        sigma_diff = torch.abs(gen_stats[1] - real_stats[1])
        constraint_sum += (mu_diff + sigma_diff)
        cnt += 1
        
    constraint_loss = constraint_sum / cnt
    constraint_loss /= mini_batch.size(1)
    return scaling_factor * constraint_loss

def distribution_fitting(config, 
                         model,
                         data):
    # Global distribution fitting
    # As described by Gong et al. in https://arxiv.org/abs/2212.01521
    # Returns a tensor of shape [2, config.latent_dim],
    # where tensor[0] is mu, and tensor[1] is sigma

    print("Applying GDF to Autoencoder...")

    initial_aggregate = (0, 0, 0) # mean, variance, samplevariance
    result_batch = [initial_aggregate] * config.latent_dim

    for batch_idx, batch in enumerate(yieldBatch(1_000, data)):
        original_lens_batch = real_lengths(batch, config.MAX_SENT_LEN)
        padded_batch = pad_batch(batch, config.MAX_SENT_LEN)
        padded_batch = torch.LongTensor(padded_batch).to(model.device)
        with torch.no_grad():
            if model.name == "variational_autoencoder":
                output = model.encoder(padded_batch, original_lens_batch)
                # extract last hidden state
                context = []
                for sequence, unpadded_len in zip(output, original_lens_batch):
                    context.append(sequence[unpadded_len-1, :])
                context = torch.stack((context))
                z = model.reparameterize(model.z_mean(context), model.z_log_var(context))
            elif model.name == "cnn_autoencoder":
                z, _ = model.encoder(padded_batch)
            elif model.name == "CNN_DCNN_WN":
                z, _ = model.encoder(padded_batch)
                z = z.squeeze(-1)
            elif model.name == "CNN_DCNN":
                z = model.encoder(padded_batch)
                z = z.squeeze(-1)
            else:
                z, _ = model.encoder(padded_batch, original_lens_batch)
            # [B, H] -> [H, B]
            z = torch.transpose(z, 1, 0)
            z = z.cpu().detach().numpy()
            # update mean, variance, samplevariance online
            for idx, hidden in enumerate(z):
                result_batch[idx] = update(result_batch[idx], hidden)

    # finalising mu and sigma
    for idx, elem in enumerate(result_batch):
        result_batch[idx] = finalize(elem)

    distribution_mean = [x[0] for x in result_batch]
    distribution_variance = [x[1] for x in result_batch]
    distribution_mean = torch.FloatTensor(distribution_mean)
    distribution_variance = torch.FloatTensor(distribution_variance)
    fitted_distribution = torch.stack((distribution_mean, distribution_variance)).to(config.device)
    return fitted_distribution