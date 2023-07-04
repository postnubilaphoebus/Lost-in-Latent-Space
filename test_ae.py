from models import AutoEncoder, CNNAutoEncoder, ExperimentalAutoencoder, VariationalAutoEncoder, CNN_DCNN_WN, CNN_DCNN
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_scoring
from rouge_score import rouge_scorer
import torch
import os
import sys
import random
import time
from utils.helper_functions import load_data_and_create_vocab, \
                                   prepare_data, \
                                   yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   average_over_nonpadded, \
                                   reformat_decoded_batch, \
                                   rouge_and_bleu, \
                                   load_vocab#,  \
                                   #return_bert_score

def load_model(config, model_name, model_path, weights_matrix = None):
    if os.path.isfile(model_path):
        if model_name == "default_autoencoder":
            if "tf" in model_path:
                print("model trained with teacher forcing")
                model = AutoEncoder(config, weights_matrix, teacher_forcing=True)
            else:
                model = AutoEncoder(config, weights_matrix)
            model.to(model.device)
        elif model_name == "cnn_autoencoder":
            if "tf" in model_path:
                print("model trained with teacher forcing")
                model = CNNAutoEncoder(config, weights_matrix, teacher_forcing=True)
            else:
                model = CNNAutoEncoder(config, weights_matrix)
            model.to(model.device)
        elif model_name == "strong_autoencoder":
            model = ExperimentalAutoencoder(config, weights_matrix)
            model = model.apply(ExperimentalAutoencoder.init_weights)
            model.to(model.device)
        elif model_name == "CNN_DCNN":
            model = CNN_DCNN(config)
            model = model.apply(CNN_DCNN.init_weights)
            model.to(config.device)
        elif model_name == "CNN_DCNN_WN":
            model = CNN_DCNN_WN(config)
            model = model.apply(CNN_DCNN_WN.init_weights)
            model.to(config.device)
        elif model_name == "variational_autoencoder":
            model = VariationalAutoEncoder(config, weights_matrix)
            model = model.apply(VariationalAutoEncoder.init_weights)
            model.to(model.device)
        else:
            sys.exit("no valid model name provided")
        model.load_state_dict(torch.load(model_path, map_location = model.device))
    else:
        sys.exit("ae model path does not exist")
    return model

def test(config):
    #location = "/data/s4184416/peregrine/saved_aes/epoch_4_model_cnn_autoencoder.pth"
    #model_path = "/Users/lauridsstockert/Desktop/test_new_models/saved_aes/epoch_50_model_CNN_DCNN_regime_normal_latent_mode_dropout.pth"
    #model_path = "/Users/lauridsstockert/Desktop/test_new_models/saved_aes/epoch_50_model_CNN_DCNN_regime_normal_gaussian_latent_mode_dropout.pth"
    #print("epoch_50_model_CNN_DCNN_regime_normal_gaussian_latent_mode_dropout.pth")
    model_path = "/Users/lauridsstockert/Desktop/test_new_models/saved_aes/epoch_5_model_default_autoencoder_tf_regime_word-mixup_latent_mode_dropout.pth"
    #model_path = location
    print("epoch_5_model_default_autoencoder_tf_regime_word-mixup_latent_mode_dropout.pth")
    model_name = "default_autoencoder"
    if model_name == "variational_autoencoder":
        config.encoder_dim = 600
    #model_name = "variational_autoencoder"
    #model_path =  location
    #model_path = "/Users/lauridsstockert/Desktop/test_new_models/saved_aes/epoch_5_model_variational_autoencoder.pth"
    model = load_model(config, model_name, model_path)
    model.eval()
    if "tf" in model_path:
        loaded_sents = 50_000
    else:
        loaded_sents = 10_000
    data = load_data_from_file("corpus_v40k_ids.txt", max_num_of_sents = loaded_sents)
    if "tf" in model_path:
        loaded_sents = 10_000
        data = data[loaded_sents:]
        print("only loaded {} sentences".format(loaded_sents))
    vocab, revvocab = load_vocab("vocab_40k.txt", 40_000)
    config.vocab_size = len(revvocab)

    scorer = rouge_scorer.RougeScorer(['rouge1', "rouge2", "rouge3", 'rouge4'], use_stemmer=True)
    score_names = ["rouge1", "rouge2", "rouge3", "rouge4", "bleu1", "bleu2", "bleu3", "bleu4", "bert score"]

    if model.name == "CNN_DCNN" or model.name == "CNN_DCNN_WN":
        config.MAX_SENT_LEN = 29

    original_lens_batch = real_lengths(data, config.MAX_SENT_LEN)
    padded_batch = pad_batch(data, config.MAX_SENT_LEN)
    
    step_size = 100
    decoded_list = []
    with torch.no_grad():
        for i in range(0, 10_000, step_size):
            if model.name != "variational_autoencoder":
                if model.name != "CNN_DCNN" and model.name != "CNN_DCNN_WN":
                    decoded_logits = model(torch.LongTensor(padded_batch[i:i+step_size]).to(model.device), original_lens_batch[i:i+step_size])
                else:
                    y, z, t, u = None, None, None, None
                    decoded_logits = model(torch.LongTensor(padded_batch[i:i+step_size]).to(model.device), y, z, t, u)
            else:
                _, _, decoded_logits = model(torch.LongTensor(padded_batch[i:i+step_size]).to(model.device), original_lens_batch[i:i+step_size])
            decoded_tokens = torch.argmax(decoded_logits, dim = -1)
            decoded_tokens = reformat_decoded_batch(decoded_tokens, 0, config.MAX_SENT_LEN)
            decoded_list.extend(decoded_tokens)
    print("done argmaxing validation data, moving to eval")

    scores = [0] * 9

    decoded_sents = []
    target_sents = []

    for decoded, target in zip(decoded_list, padded_batch):
        try:
            first_zero_tgt = target.index(0)
            target = target[:first_zero_tgt]
            try:
                lowest_idx = min(decoded.index(1), decoded.index(0)) 
                decoded = decoded[:lowest_idx]
            except:
                pass
        except:
            try:
                lowest_idx = min(decoded.index(1), decoded.index(0)) 
                decoded = decoded[:lowest_idx]
            except:
                pass
        dec_sent = [revvocab[x] for x in decoded]
        target_sent = [revvocab[x] for x in target]
        dec_sent = " ".join(dec_sent)
        target_sent = " ".join(target_sent)
        decoded_sents.append(dec_sent)
        target_sents.append(target_sent)
        interim = rouge_and_bleu(dec_sent, target_sent, scorer)
        for i in range(len(interim)):
            scores[i] += interim[i]

    #batched_score = 0
    #cnt = 0
    #for i in range(0, 10_000, step_size):
        #dec = decoded_sents[i:i+1000]
        #tar = target_sents[i:i+1000]
        #bs = return_bert_score(dec, tar, device=config.device, batch_size=step_size)
        #bs = round(bs, 4)
        #batched_score += bs
        #cnt +=1
    scores = [x / loaded_sents for x in scores]
    scores[-1] = 0 #batched_score / cnt
    print("bert score is 0 by default because inactive")
    for score, name in zip(scores, score_names):
        print("{}: {}".format(name, score))