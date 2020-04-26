from scripts.util import write_rows_csv, return_ind_for_word_list, read_word2_ind, clean_sentence
from scripts.util import is_there_highlight_first, is_there_highlight_last
import re
import csv
word2ind = read_word2_ind('common')

#TODO clean
def get_cleaned_triplets():
    questions = []
    pa1 = []
    pa2 = []
    with open('data/highlist_triples.csv') as f:
        reader = csv.reader(f, delimiter='\t')
        print_count = 0
        for row in reader:
            print_count+=1
            if print_count%1000 == 0:
                print('So far ', print_count)
            q = clean_sentence(row[0])
            p1 = clean_sentence(row[1]).split(' ')
            p2 = clean_sentence(row[2]).split(' ')
            add = True
            p1_has_snippet_f = is_there_highlight_first(' '.join(p1))
            p2_has_snippet_f = is_there_highlight_first(' '.join(p2))
            p1_has_snippet_l = is_there_highlight_last(' '.join(p1))
            p2_has_snippet_l = is_there_highlight_last(' '.join(p2))

            if len(p1) > 350:
                if p1_has_snippet_f:
                    p1 = ' '.join(p1[:351])
                elif p1_has_snippet_l:
                    p1 = ' '.join(p1[-351:])
                else:
                    add = False
            else:
                p1 = ' '.join(p1)
            if len(p2) > 350:
                if p2_has_snippet_f:
                    p2 = ' '.join(p2[:351])
                elif p2_has_snippet_l:
                    p2 = ' '.join(p2[-351:])
                else:
                    add = False
            else:
                p2 = ' '.join(p2)
            if add:
                questions.append(q)
                pa1.append(p1)
                pa2.append(p2)
    return questions, pa1, pa2


questions, pa1, pa2 = get_cleaned_triplets()
print(questions[0])
X_Q = []
X_pa1 = []
X_pa2 = []
for x in questions:
    X_Q.append(return_ind_for_word_list(x, word2ind))
for x,y in zip(pa1, pa2):
    X_pa1.append(return_ind_for_word_list(x, word2ind))
    X_pa2.append(return_ind_for_word_list(y, word2ind))

write_rows_csv('data/common/question_to_ids.csv', X_Q)
write_rows_csv('data/common/pa1_to_ids.csv', X_pa1)
write_rows_csv('data/common/pa2_to_ids.csv', X_pa2)
