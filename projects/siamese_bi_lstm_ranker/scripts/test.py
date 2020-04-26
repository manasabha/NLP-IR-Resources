import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.util import read_files_and_return_dataset
from scripts.constant_model_params import device

def get_original_loss(model_name, q_encoder, p_encoder, dataloader, dataset_type, attn_model=None):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    batch_to_data_q = {}
    batch_to_data_pa1 = {}
    batch_to_data_pa2 = {}
    count = 0
    total_loss = 0
    for i, (q, pa1, pa2) in enumerate(dataloader):
        count += len(q)
        q_enc_out = q_encoder(q, batch_to_data_q, i, device, attn_model)
        pa1_enc_out = p_encoder(pa1, batch_to_data_pa1, i, device,
                                q_enc_out['encoded_question_vector'], attn_model)
        pa2_enc_out = p_encoder(pa2, batch_to_data_pa2, i, device,
                                q_enc_out['encoded_question_vector'], attn_model)

        cos_q_p1 = cos(q_enc_out['encoded_question_vector'],
                       pa1_enc_out['encoded_passage_vector']).to(device)
        cos_q_p2 = cos(q_enc_out['encoded_question_vector'],
                       pa2_enc_out['encoded_passage_vector']).to(device)
        loss = torch.log(1 + torch.exp(10 * (cos_q_p2 - cos_q_p1))).sum(0).to(device)
        total_loss += loss.item()

    print('for model {} Total loss for {} data  {} norm loss {}  len {}'.format(model_name, dataset_type,
                                                                                total_loss, total_loss/count, count))

def get_loss_function_value():
    from scripts.util import np_collate, load_models
    from scripts.constant_model_params import params_to_load_models
    train_model_name = input('Enter train model name to be used: ')
    val_model_name = input('Enter val model name to be used: ')
    load_datasets = int(input('Load saved datsets 0/1: '))
    train_loaded_models = load_models(train_model_name, device, params_to_load_models, is_eval=True)
    val_loaded_models = load_models(val_model_name, device, params_to_load_models,  is_eval=True)

    bs = 1024
    if load_datasets:
        train_dataset = torch.load('data/train_dataset')
        train_dataloader = DataLoader(train_dataset, collate_fn=np_collate, batch_size=bs)
        val_dataset = torch.load('data/val_dataset')
        val_dataloader = DataLoader(val_dataset, collate_fn=np_collate, batch_size=bs)
    else:
        triple_set_train, triple_set_val = read_files_and_return_dataset(is_shuffle=False)
        train_dataloader = DataLoader(dataset=triple_set_train, batch_size=bs, collate_fn=np_collate)
        val_dataloader = DataLoader(dataset=triple_set_val, batch_size=bs, collate_fn=np_collate)

    get_original_loss(train_model_name, train_loaded_models['question_encoder'], train_loaded_models['passage_encoder'],
                      train_dataloader, 'train', train_loaded_models['attention_model'])
    get_original_loss(val_model_name, val_loaded_models['question_encoder'], val_loaded_models['passage_encoder'],
                      val_dataloader, 'val', val_loaded_models['attention_model'])

get_loss_function_value()