from scripts.util import get_length_of_dataset, read_ind2vec, date_as_float
import torch
hyper_params = {
    'emb_size': 300,
    'nh': 100,
    'nl': 1,
    'ip_size': get_length_of_dataset('common'),
    'linear_units': 200,
    'out_em_size': 512,
    'ind2vec': {},#read_ind2vec('common'),
    'attn_units': 100,
    'ndir': 2
}
params = {
    'enable_attention': True,
    'use_common_attention_model': False,
    'save_weights': True,
    'use_question_context': False,
    'return_context': False,
}
BUCKET_NAME = 'manasa-mscac-project'
lr = 0.01
momentum = 0.9
input_fields_with_default_values = {
    'bs': 256,
    'text': 'attn_biLSTM_'+date_as_float(),
}
epochs = 201
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training will happen on {}'.format(device))

params_to_load_models = {
    'enable_attention': True,
    'use_common_attention_model': False,
    'save_weights': True,
    'use_question_context': False,
    'return_context': False
}

params.update(hyper_params)
params_to_load_models.update(hyper_params)