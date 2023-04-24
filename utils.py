# coding: UTF-8
import os
import datetime

import numpy
import pandas as pd
import torch
import numpy as np
import random
import pickle as pkl
from tqdm import tqdm
import math
import time
from datetime import timedelta


CLS, SEP = '[CLS]', '[SEP]'

def set_random_state(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure the results can be replicated
    torch.backends.cudnn.deterministic = True

def build_branch(df, now_id, branch):
    temp_df = df.loc[df['id'] == now_id]
    if len(temp_df) == 0: # error
        return branch
    branch.append(now_id)
    if temp_df['is_post'].iloc[0] == True:
        return branch #is top node
    else:
        next_id = temp_df['replied_to_tweet_id'].iloc[0]
        if len(df.loc[df['id'] == next_id]) > 0:
            #next_round
            branch = build_branch(df, next_id, branch)
        else:
            branch.append(temp_df['conversation_id'].iloc[0])
            return branch
    return branch

def check_error(df, i, branch):
    if  df['replied_to_tweet_id'].iloc[i] != 0:
        parent = df['replied_to_tweet_id'].iloc[i]
        if parent != branch[-2]:
            print("found_error: ", branch[-2], parent)
            df['replied_to_tweet_id'].iloc[i] = branch[-2]
    return df

def get_branch(df):
    print('building branches...')
    for idx in tqdm(range(len(df))):
        branch = []
        now_id = df['id'].iloc[idx]
        branch = build_branch(df, now_id, branch)
        branch.reverse()
        df = check_error(df, idx, branch)
        df['branch'].iloc[idx] = branch
    return df

def get_sec_diff(start_time, end_time):
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    seconds = int((end_time - start_time).total_seconds())
    return seconds

def get_abs_pos(df, pad_size = 101, d_model = 384):
    pe = torch.zeros(pad_size, d_model)
    timestamps = df['create_time'].to_list()
    t0 = timestamps[0]

    time_diffs = [get_sec_diff(t0, t1) for t1 in timestamps]
    # Create a list of infinite values to use for padding
    inf_secs = get_sec_diff('1900-1-1 00:00:00', '2023-12-23 00:00:00')
    infinite_values = [inf_secs] * (pad_size - len(time_diffs))

    padded_time_diffs = time_diffs + infinite_values
    time_diffs = torch.tensor(padded_time_diffs)
    position = time_diffs.unsqueeze(1)


    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    pe = pe.squeeze(1)
    pe = pe.cpu().numpy().tolist()
    return pe

def get_input(df, pad_size = 101, d_model = 384):
    embs = []
    for idx in range(len(df)):
        emb = df['embedding'].iloc[idx] #numpy
        embs.append(emb)
    embs = np.array(embs)
    paddings = np.zeros((pad_size - len(df), d_model))
    padded_embs = np.concatenate([embs, paddings], axis=0)
    padded_embs = padded_embs.tolist()
    return padded_embs

def get_theta_indexs_and_mask(df, pad_size = 101):
    pad = [0,0]
    same_branch = [0, 1]
    diff_branch = [1, 0]
    values = [[pad] * pad_size] * pad_size
    mask = [[0] * pad_size] * pad_size
    for i in range(len(df)):
        t1 = df['id'].iloc[i]
        branch_1 = df['branch'].iloc[i]
        for j in range(len(df)):
            branch_2 = df['branch'].iloc[j]
            if (set(branch_1) <= set(branch_2)) or (set(branch_2) <= set(branch_1)):
                values[i][j] = same_branch
                values[j][i] = same_branch
            else:
                values[i][j] = diff_branch
                values[j][i] = diff_branch
            mask[i][j] = 1
            mask[j][i] = 1
    return values, mask



def build_dataset(config):
    '''build train, dev, test set'''
    def load_dataset(path, mode = "train"):

        df = pd.read_parquet(path)
        conv_ids = df['conversation_id'].unique()
        if mode == "train":
            df['label'] = df['label'].fillna(2)
            df['label'] = df['label'].map({'FAVOR':0,'AGAINST':1, 2: 2})
            pro_num = len(df.loc[df['label'] == 0])
            anti_num = len(df.loc[df['label'] == 1])
            weights = torch.Tensor([anti_num / (pro_num + anti_num), pro_num / (pro_num + anti_num)]).to(config.device)
            config.weights = weights
            print("using weights: ", config.weights)
            random.seed(1126)
            random.shuffle(conv_ids)
            conv_ids = conv_ids[:2500]
        elif mode == "dev":
            df['label'] = df['label'].fillna(2)
            df['label'] = df['label'].map({'FAVOR': 0, 'AGAINST': 1, 2: 2})
            conv_ids = df['conversation_id'].unique()
            random.shuffle(conv_ids)
            conv_ids = conv_ids[-500:]
        else:
            df['label'] = df['label'].map({'FAVOR': 0, 'AGAINST': 1, 'NEITHER': 2})

        Inputs = []
        filename = config.dataset_path + '/' + mode + '_Inputs.pkl'
        if os.path.exists(filename):
            print("reading from cache: ", filename)
            fileObject = open(filename, 'rb')
            Inputs = pkl.load(fileObject)
            fileObject.close()
        else:
            print("no cache, constructing")
            '''create features'''
            for conversation_id in tqdm(conv_ids):
                conv = df.loc[df['conversation_id'] == conversation_id]
                if len(conv) > 101:
                    conv = conv.head(101)
                conv_size = len(conv)
                label = conv.label.to_list()
                label = [int(x) for x in label]
                if set(label) == set([2]):
                    continue
                padding_label = [2] * (101 - conv_size)
                label.extend(padding_label)
                conv_embs = get_input(conv)
                conv_poss = get_abs_pos(conv)
                theta_indexs, mask = get_theta_indexs_and_mask(conv)

                input = (conv_embs, conv_poss, theta_indexs, mask, label)
                Inputs.append(input)

            fileObject = open(filename, 'wb')
            pkl.dump(Inputs, fileObject)
            fileObject.close()
        return Inputs

    train = load_dataset(config.train_path)
    print("train_size:", len(train))
    dev = load_dataset(config.train_path, mode = "dev")
    print("dev_size:", len(dev))
    test = load_dataset(config.test_path, mode = "test")
    print("test_size:", len(test))
    return train, dev, test


class DatasetIterator(object):
    '''Dataset Iterator to generate mini-batches for model training.
    Params:
        batches: input dataset
        batch_size: size of mini-batches
        device: computing device
    '''

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # record if number of batches is an int
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, data):
        '''convert data to tensor '''
        conv_embs = torch.Tensor([_[0] for _ in data]).to(self.device)
        conv_poss = torch.Tensor([_[1] for _ in data]).to(self.device)
        theta_indexs = torch.LongTensor([_[2] for _ in data]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in data]).to(self.device)
        y = torch.LongTensor([_[4] for _ in data]).to(self.device)
        return (conv_embs, conv_poss, theta_indexs, mask), y

    def __next__(self):
        '''get next batch'''
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index *
                                   self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index *
                                   self.batch_size: (self.index +
                                                     1) *
                                   self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    '''API for building dataset iterator'''
    iterator = DatasetIterator(dataset, config.batch_size, config.device)
    return iterator

def get_time_dif(start_time):
    """compute time difference"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
