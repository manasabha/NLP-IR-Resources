from gensim.models import KeyedVectors
import csv
from scripts.util import clean_text, read_word2_ind, get_DRMM_vecs_for_ind, write_all_word_ind_vec_to_files_common


def parse_and_add_to_dict(word2ind, ind2word, count, orig_w2i):
    for word in orig_w2i:
        clean_word = clean_text(word)
        if clean_word not in word2ind:
            word2ind[clean_word] = count
            ind2word[count] = clean_word
            count+=1
    return count

WORD2VEC = 'data/_W2V_GENSIM_ROSS_News_Dim_300_Window_5_MinCount_75'

wv_from_text = KeyedVectors.load_word2vec_format(WORD2VEC)

word2ind_cur_Q = read_word2_ind('Q')
word2ind_cur_P = read_word2_ind('P')


word2ind_full = {'pad': 0}
ind2word_full = {0: 'pad'}
count = 1
count = parse_and_add_to_dict(word2ind_full, ind2word_full, count, word2ind_cur_Q)
count = parse_and_add_to_dict(word2ind_full, ind2word_full, count, word2ind_cur_P)
count = parse_and_add_to_dict(word2ind_full, ind2word_full, count, wv_from_text.vocab)

print('Total number of words {}'.format(len(word2ind_full)))

ind2vec_full = get_DRMM_vecs_for_ind(word2ind_full, wv_from_text)
write_all_word_ind_vec_to_files_common(ind2vec_full, ind2word_full, word2ind_full)
