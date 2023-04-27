# coding: UTF-8
import numpy as np
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import spacy


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level

    def load_dataset(path, pad_size = config.pad_size, train = False):
        contents = []
        nlp = spacy.load("en_core_web_md")
        pro_num = 0
        anti_num = 0
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                if train == True:
                    if label == '0':
                        pro_num+=1
                    else:
                        anti_num+=1
                tokens = nlp(content)
                embs = [token.vector for token in tokens] #(300, len(token))
                embs = np.array(embs)

                t_tokens = nlp("vaccination")
                t_embs = np.array([token.vector for token in t_tokens])

                seq_len = len(tokens)
                if pad_size:
                    if len(tokens) < pad_size:
                        paddings = np.zeros((pad_size - seq_len, 300))
                        final_embs = np.concatenate((embs,paddings), axis=0)
                    else:
                        final_embs = embs[:pad_size, :]
                final_embs = final_embs.tolist()
                t_embs = t_embs.tolist()
                contents.append((final_embs, int(label), seq_len, t_embs))
        if not train:
            #weights = torch.Tensor([anti_num / (pro_num + anti_num), pro_num / (pro_num + anti_num)]).to(config.device)
            #config.weights = weights
            config.weights = torch.Tensor([0.4,0.6]).to(config.device)
            print("using weights: ", config.weights)
        return contents


    train = load_dataset(config.train_path, config.pad_size, train=True)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.Tensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        t = torch.Tensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, t), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
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
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
