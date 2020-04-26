from torch.utils.data import Dataset
import numpy as np

class TripleDataset(Dataset):
    def __init__(self, question_set, pa1_set, pa2_set, rand_inds, split=0.8, is_train=False):
        self.random_ind = rand_inds
        self.split_ind = int(split*len(question_set))
        print('Total len {} split {}'.format(len(question_set), self.split_ind))
        print('First 10 random inds ',self.random_ind[:10])
        if is_train:
            self.inds = self.random_ind[:self.split_ind]
        else:
            self.inds = self.random_ind[self.split_ind:]

        self.questions = np.array(question_set)[self.inds].tolist()
        self.better_passages = np.array(pa1_set)[self.inds].tolist()
        self.passages = np.array(pa2_set)[self.inds].tolist()
    def __len__(self):
        return len(self.questions)
    def __getitem__(self,ind):
        return self.questions[ind], self.better_passages[ind], self.passages[ind]

# add actual preprocessing code.

class TripleDatasetNoShuffle(Dataset):
    def __init__(self, question_set, pa1_set, pa2_set, split=0.8, is_train=False):
        self.split_ind = int(split*len(question_set))
        print('Total len {} split {}'.format(len(question_set), self.split_ind))
        if is_train:
            self.min_ind = 0
            self.max_ind = self.split_ind
        else:
            self.min_ind = self.split_ind
            self.max_ind = len(question_set)
        self.questions = question_set[self.min_ind:self.max_ind]
        self.better_passages = pa1_set[self.min_ind:self.max_ind]
        self.passages = pa2_set[self.min_ind:self.max_ind]
    def __len__(self):
        return len(self.questions)
    def __getitem__(self,ind):
        return self.questions[ind], self.better_passages[ind], self.passages[ind]