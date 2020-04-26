import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math

#Code from fast ai dl2 - seq2seq translation. didn't rework, as it is small and good enough
def rand_t(*sz): return torch.randn(sz)/math.sqrt(sz[0])
def rand_p(*sz): return nn.Parameter(rand_t(*sz))

#Rest of this project is my code except those 3 functions.
class Attention(nn.Module):
    def __init__(self, attn_units):
        super(Attention, self).__init__()
        self.attn_units = attn_units
        self.attn_weights = rand_p(self.attn_units)

    def forward(self, capped_lstm_out):
        attn_layer_out = capped_lstm_out @ self.attn_weights
        attn_softmax = F.softmax(attn_layer_out, 0)
        return attn_softmax

class BaseEncoder(nn.Module):
    def __init__(self, emb_size, nh, nl, ip_size, linear_units, attn_units,
                 out_em_size, ind2vec, ndir, enable_attention,
                 use_common_attention_model, save_weights=False):
        super(BaseEncoder, self).__init__()
        self.emb_size = emb_size
        self.nh = nh
        self.nl = nl
        self.ip_size = ip_size
        self.linear_units = linear_units
        self.out_em_size = out_em_size
        self.ndir = ndir
        self.bidir = ndir == 2
        self.ldim = self.nh * self.ndir * self.nl
        self.enable_attention = enable_attention

        self.in_emb_layer = self.create_emb(self.ip_size, ind2vec, self.emb_size)
        self.dropout = nn.Dropout(0.20)
        self.lstm_layer = nn.LSTM(hidden_size=self.nh, input_size=self.emb_size,
                                  num_layers=self.nl, batch_first=True,
                                  bidirectional=self.bidir)
        if self.enable_attention:
            self.attn_units = attn_units
            self.interim_layer = nn.Linear(self.ldim, self.attn_units)
            self.use_common_attention_model = use_common_attention_model
            self.save_weights = save_weights
            if not self.use_common_attention_model:
                self.attn_weights = rand_p(self.attn_units)

        self.linear_layer = nn.Linear(self.ldim, self.linear_units)
        self.out_emb_layer = nn.Linear(self.linear_units, self.out_em_size)

    def create_emb(self, vocab_size, ind2vec, em_sz):
        # Code from fast ai dl2 - seq2seq translation. didn't rework, as it is small and good enough
        emb = nn.Embedding(vocab_size, em_sz, padding_idx=1)
        wgts = emb.weight.data
        miss = []
        for i in range(vocab_size):
            try:
                wgts[i] = torch.from_numpy(ind2vec[i])
            except:
                miss.append(i)
        print('No of words with no pretrained embeddings {}'.format(len(miss)))
        return emb

    def get_seq_lengths(self, input_tensor):
        lens = []
        for x in input_tensor:
            count = len(x) - (x == 0).sum(0)
            lens.append(count)
        return torch.LongTensor(lens)

    def update_batch_dict(self, input, batch_info_dict, batch_no):
        batch_info_dict[batch_no] = {}
        batch_info_dict[batch_no]['lens_sorted'], batch_info_dict[batch_no]['sorted_ind'] = \
            torch.sort(self.get_seq_lengths(input), descending=True)
        _, batch_info_dict[batch_no]['ind_retrieve_orig'] = torch.sort(batch_info_dict[batch_no]['sorted_ind'])


class QuestionEncoder(BaseEncoder):
    def __init__(self, emb_size, nh, nl, ip_size, linear_units, attn_units,
                 out_em_size, ind2vec, ndir, enable_attention, return_context,
                 use_common_attention_model, save_weights=False):

        super(QuestionEncoder, self).__init__(emb_size, nh, nl, ip_size, linear_units, attn_units,
                                              out_em_size, ind2vec, ndir, enable_attention,
                                              use_common_attention_model, save_weights)
        if self.enable_attention:
            self.return_context = return_context

    def question_attention(self, lstm_out_packed, attn_model, device):
        return_dict = {
            'lstm_attn_out': None,
            'attn_weights': None,
            'context_vector': None
        }

        lstm_out, _ = pad_packed_sequence(lstm_out_packed)
        capped_lstm_out = torch.tanh(self.interim_layer(lstm_out))
        if self.use_common_attention_model:
            attn_softmax = attn_model(capped_lstm_out.to(device))
        else:
            attn_layer_out = capped_lstm_out @ self.attn_weights
            attn_softmax = F.softmax(attn_layer_out, 0)

        lstm_out_weighted = (attn_softmax.unsqueeze(2)*lstm_out).sum(0)
        return_dict['lstm_attn_out'] = lstm_out_weighted
        if self.save_weights:
            return_dict['attn_weights'] = attn_softmax
        if self.return_context:
            return_dict['context_vector'] = lstm_out_weighted

        return return_dict

    def forward(self, x, batch_info_dict, batch_no, device, attn_model=None, save_weights=False):
        output_dict = {
            'encoded_question_vector': None,
            'attn_weights': None,
            'context_vector': None
        }

        h0 = torch.zeros(self.nl * self.ndir, x.size(0), self.nh).to(device)
        c0 = torch.zeros(self.nl * self.ndir, x.size(0), self.nh).to(device)

        emb = self.dropout(self.in_emb_layer(x))
        if batch_no not in batch_info_dict:
            self.update_batch_dict(x, batch_info_dict, batch_no)

        emb = emb[batch_info_dict[batch_no]['sorted_ind'].to(device)]
        emb = pack_padded_sequence(emb, batch_info_dict[batch_no]['lens_sorted'], batch_first=True)

        lstm_out_packed, h1 = self.lstm_layer(emb, (h0, c0))
        if self.enable_attention:
            attention_result_dict = self.question_attention(lstm_out_packed, attn_model, device)
            if not (attention_result_dict['attn_weights'] is None):
                output_dict['attn_weights'] = attention_result_dict['attn_weights']
            if not (attention_result_dict['context_vector'] is None):
                output_dict['context_vector'] = attention_result_dict['context_vector']
            lstm_out = attention_result_dict['lstm_attn_out']

        else:
            lstm_out = h1

        lin_out = self.linear_layer(self.dropout(lstm_out.view(-1, self.ldim)))
        out_emb = self.out_emb_layer(lin_out).view(-1, self.out_em_size)
        out_emb = out_emb[batch_info_dict[batch_no]['ind_retrieve_orig'].to(device)]

        output_dict['encoded_question_vector'] = out_emb

        return output_dict

class PassageEncoder(BaseEncoder):
    def __init__(self, emb_size, nh, nl, ip_size, linear_units, attn_units,
                 out_em_size, ind2vec,ndir, enable_attention, use_question_context,
                 use_common_attention_model, save_weights=False):

        super(PassageEncoder, self).__init__(emb_size, nh, nl, ip_size, linear_units, attn_units,
                                             out_em_size, ind2vec, ndir, enable_attention,
                                             use_common_attention_model, save_weights)
        if self.enable_attention:
            self.use_question_context = use_question_context
            if self.use_question_context:
                # Concatenate question context to passage attn
                self.linear_layer_context = nn.Linear(2 * self.ldim, self.ldim)


    def passage_attention(self, lstm_out_packed, attn_model, encoded_question, device):

        return_dict = {
            'lstm_attn_out': None,
            'attn_weights': None,
        }

        lstm_out, _ = pad_packed_sequence(lstm_out_packed)
        capped_lstm_out = torch.tanh(self.interim_layer(lstm_out))

        if self.use_common_attention_model:
            attn_softmax = attn_model(capped_lstm_out.to(device))
        else:
            attn_layer_out = capped_lstm_out @ self.attn_weights
            attn_softmax = F.softmax(attn_layer_out, 0)

        lstm_out_weighted = (attn_softmax.unsqueeze(2) * lstm_out).sum(0)
        lstm_out_weighted = lstm_out_weighted.view(-1, self.ldim)
        return_dict['lstm_attn_out'] = lstm_out_weighted

        if self.use_question_context:
            combined_context = torch.cat([lstm_out_weighted.to(device), encoded_question.to(device)], 1).to(device)
            lin_context_out = self.linear_layer_context(combined_context)
            return_dict['lstm_attn_out'] = lin_context_out

        if self.save_weights:
            return_dict['attn_weights'] = attn_softmax
        return return_dict


    def forward(self, x, batch_info_dict, batch_no, device, encoded_question, attn_model=None):
        output_dict = {
            'encoded_passage_vector': None,
            'attn_weights': None
        }

        h0 = torch.zeros(self.nl * self.ndir, x.size(0), self.nh).to(device)
        c0 = torch.zeros(self.nl * self.ndir, x.size(0), self.nh).to(device)
        emb = self.dropout(self.in_emb_layer(x))
        if batch_no not in batch_info_dict:
            self.update_batch_dict(x, batch_info_dict, batch_no)

        emb = emb[batch_info_dict[batch_no]['sorted_ind'].to(device)]
        emb = pack_padded_sequence(emb, batch_info_dict[batch_no]['lens_sorted'], batch_first=True)

        lstm_out_packed, h1 = self.lstm_layer(emb, (h0, c0))

        if self.enable_attention:
            attention_result_dict = self.passage_attention(lstm_out_packed, attn_model, encoded_question, device)
            if not (attention_result_dict['attn_weights'] is None):
                output_dict['attn_weights'] = attention_result_dict['attn_weights']
            lstm_out = attention_result_dict['lstm_attn_out']

        else:
            lstm_out = h1

        lin_out = self.linear_layer(self.dropout(lstm_out.view(-1, self.ldim)))

        out_emb = self.out_emb_layer(lin_out).view(-1, self.out_em_size)
        out_emb = out_emb[batch_info_dict[batch_no]['ind_retrieve_orig'].to(device)]
        output_dict['encoded_passage_vector'] = out_emb
        return output_dict
