import time
import matplotlib.pyplot as plt
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scripts.constant_model_params import params, lr, momentum, input_fields_with_default_values, epochs, BUCKET_NAME
from scripts.constant_model_params import device, params_to_load_models
from scripts.util import read_files_and_return_dataset, get_user_input, np_collate, upload_file_to_s3
from scripts.util import instantiate_models, validate_user_choices
import os


def measure_val_loss(triple_loader_val, q_enc, p_enc, attn, cos):
    total_loss_val = 0
    batch_to_val_data_q = {}
    batch_to_val_data_pa1 = {}
    batch_to_val_data_pa2 = {}
    start = time.time()
    val_models = instantiate_models(params_to_load_models, device)
    new_q_enc = val_models['q_enc']
    new_p_enc = val_models['p_enc']
    new_q_enc.load_state_dict(q_enc.state_dict())
    new_p_enc.load_state_dict(p_enc.state_dict())
    new_q_enc.eval()
    new_p_enc.eval()

    if params['use_common_attention_model']:
        attn_model = val_models['attn_model']
        attn_model.load_state_dict(attn.state_dict())
        attn_model.eval().to(device)
    else:
        attn_model = None

    for j, (q_val, pa1_val, pa2_val) in enumerate(triple_loader_val):
        if j % 1000 == 0:
            print('So far {} batches'.format(j))
        question_model_output_val = new_q_enc(q_val, batch_to_val_data_q, j, device, attn_model)
        passage_model_output1_val = new_p_enc(pa1_val, batch_to_val_data_pa1, j, device,
                                              question_model_output_val['context_vector'], attn_model)
        passage_model_output2_val = new_p_enc(pa2_val, batch_to_val_data_pa2, j, device,
                                              question_model_output_val['context_vector'], attn_model)

        cos_q_p1_val = cos(question_model_output_val['encoded_question_vector'],
                           passage_model_output1_val['encoded_passage_vector']).to(device)
        cos_q_p2_val = cos(question_model_output_val['encoded_question_vector'],
                           passage_model_output2_val['encoded_passage_vector']).to(device)

        loss_val = torch.log(1 + torch.exp(10 * (cos_q_p2_val - cos_q_p1_val))).sum(0).to(device)
        total_loss_val += loss_val.item()

        del (cos_q_p1_val)
        del (cos_q_p2_val)
        del(loss_val)
        torch.cuda.empty_cache()
    del(new_p_enc)
    del(new_q_enc)
    del(attn_model)
    end = time.time()
    print('Took {} for val set and val loss {}'.format(((end - start) / 60.0), total_loss_val))
    return total_loss_val

def plot_losses(train_loss, val_loss, model_name):
    plt.figure(figsize=(10,10))
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('data/loss_graphs/'+model_name)

def save_model(model, model_local_path, model_s3_path, model_type):
    full_model_local_path = 'models/'+model_type+'/'+model_local_path
    torch.save(model, full_model_local_path)
    upload_file_to_s3(full_model_local_path, model_s3_path, BUCKET_NAME, 'models/'+model_type)
    os.remove(full_model_local_path)
    torch.cuda.empty_cache()

def train():

    user_input_fields_values = get_user_input(input_fields_with_default_values)
    bs = int(user_input_fields_values['bs'])
    text = user_input_fields_values['text']

    triple_set_train, triple_set_val = read_files_and_return_dataset(device, split=0.7, is_common=True)
    triple_loader_train = DataLoader(dataset=triple_set_train, batch_size=bs, collate_fn=np_collate)
    triple_loader_val = DataLoader(dataset=triple_set_val, batch_size=bs, collate_fn=np_collate)

    train_loss = []
    val_loss = []

    train_models = instantiate_models(params, device)
    q_enc = train_models['q_enc']
    p_enc = train_models['p_enc']
    optimizer = optim.SGD(list(q_enc.parameters()) + list(p_enc.parameters()), lr=lr, momentum=momentum)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    if not validate_user_choices(params):
        print('Follow the instructions provided and rerun the script')

    if params['enable_attention'] and params['use_common_attention_model']:
        attn_model = train_models['attn_model']
        optimizer = optim.SGD(list(q_enc.parameters()) + list(p_enc.parameters()) + list(attn_model.parameters()),
                              lr=lr, momentum=momentum)
    else:
        attn_model = None

    print('Question Encoder Model')
    print(q_enc)
    print('Passage Encoder Model')
    print(p_enc)
    try:
        start_train = time.time()
        batch_to_train_data_q = {}
        batch_to_train_data_pa1 = {}
        batch_to_train_data_pa2 = {}
        model_name = 'model_bs_{}_epoch_{}_{}'.format(bs, epochs, text)

        for epoch in range(epochs):
            total_loss = 0
            start = time.time()
            for i, (q, pa1, pa2) in enumerate(triple_loader_train):
                if i % 1000 == 0:
                    print('So far {} batches'.format(i))

                question_model_output = q_enc(q, batch_to_train_data_q, i, device, attn_model)
                passage_model_output1 = p_enc(pa1, batch_to_train_data_pa1, i, device,
                                              question_model_output['context_vector'], attn_model)
                passage_model_output2 = p_enc(pa2, batch_to_train_data_pa2, i, device,
                                              question_model_output['context_vector'], attn_model)

                cos_q_p1 = cos(question_model_output['encoded_question_vector'],
                               passage_model_output1['encoded_passage_vector']).to(device)
                cos_q_p2 = cos(question_model_output['encoded_question_vector'],
                               passage_model_output2['encoded_passage_vector']).to(device)

                loss = torch.log(1 + torch.exp(10 * (cos_q_p2 - cos_q_p1))).sum(0).to(device)
                total_loss += loss.item()
                del (cos_q_p1)
                del (cos_q_p2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            train_loss.append(total_loss)
            end = time.time()
            print('Loss at epoch {} is {} took {} time'.format(epoch, total_loss, ((end - start) / 60.0)))
            val_loss.append(measure_val_loss(triple_loader_val, q_enc, p_enc, attn_model, cos))

            if epoch and epoch%1 == 0:
                print('Inside save block')
                PATH = 'model_bs_{}_epoch_{}_{}'.format(bs, epoch, text)
                s3_file = 'model_bs_{}_epoch_{}_{}'.format(bs, epoch, text)
                save_model(q_enc, PATH, s3_file, 'Q')
                save_model(p_enc, PATH, s3_file, 'P')
                if attn_model:
                    save_model(attn_model, PATH, s3_file, 'attn')
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                plot_losses(train_loss, val_loss, model_name)
            torch.cuda.empty_cache()
        end_train = time.time()
        print('Time taken for full train for {} epochs is {}'.format(epochs, ((end_train - start_train) / 60.0)))

        torch.cuda.empty_cache()
        plot_losses(train_loss, val_loss, model_name)
    except Exception as e:
        print('Final exception ', e)
        print(traceback.format_exc())
        del (q_enc)
        del (p_enc)
        del (optimizer)
        del (loss)
        del (cos)
        del(attn_model)
        print('Failed for bs {} at {}'.format(bs, i))
        torch.cuda.empty_cache()


train()
