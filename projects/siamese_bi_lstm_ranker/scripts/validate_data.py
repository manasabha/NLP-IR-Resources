import torch
from torch.utils.data import DataLoader
from scripts.util import return_word_for_ind_list_batch, read_ind2word, load_id_to_lpo_data
from scripts.util import get_lpo_data_point_for_questions, clean_sentence, similar, np_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ind2word_Q = read_ind2word('Q')
ind2word_P = read_ind2word('P')
lpo_data = load_id_to_lpo_data()

val_dataset = torch.load('data/val_dataset')
val_dataloader = DataLoader(val_dataset, collate_fn=np_collate, batch_size=10)

break_point = 1

for i,(q,pa1, pa2) in enumerate(val_dataloader):
    question_words = return_word_for_ind_list_batch(q, ind2word_Q)
    print(get_lpo_data_point_for_questions(question_words, lpo_data))
    better_passages = return_word_for_ind_list_batch(pa1, ind2word_P)
    regular_passages = return_word_for_ind_list_batch(pa2, ind2word_P)
    print(list(zip(question_words, better_passages, regular_passages)))
    if i+1>=break_point:
        break

def get_quality_pairs(loader):
    quality_pairs = []
    for i, (q, pa1, pa2) in enumerate(loader):
        question_words = return_word_for_ind_list_batch(q, ind2word_Q)
        entry_list = get_lpo_data_point_for_questions(question_words, lpo_data)
        better_passages = return_word_for_ind_list_batch(pa1, ind2word_P)
        regular_passages = return_word_for_ind_list_batch(pa2, ind2word_P)
        for i,entry_tup in enumerate(entry_list):
            entry = entry_tup[1]
            for ans in entry['answers']:
                cleaned_ans = clean_sentence(ans['ans'])
                if similar(cleaned_ans, better_passages[i])> 0.9:
                    bqua = ans['qua']
                elif similar(cleaned_ans, regular_passages[i])>0.9:
                    rqua = ans['qua']
                    break
            quality_pairs.append((bqua,rqua))
        break
    return quality_pairs

print(get_quality_pairs(val_dataloader))