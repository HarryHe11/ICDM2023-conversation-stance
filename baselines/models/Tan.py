# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [0,1]                                                      # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.weights = torch.tensor([1.0, 1.0])
        self.dropout = 0.1                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4                                      # 学习率
        self.embed = 300           # 字向量维度
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 128

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(2 * config.embed))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * config.num_layers, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x_emb = x[0]
        target_emb = x[-1] # [batch_size, T_Seq_len, embedding]
        #emb = self.embedding(x)  # [batch_size, seq_len, embedding]=[128, 32, 300]
        H, _ = self.lstm(x_emb)  # [batch_size, seq_len, hidden_size * num_direction]
        target_emb = torch.mean(target_emb, dim = 1) # [batch_size, embedding]
        target_emb = target_emb.unsqueeze(1) # [batch_size, 1, embedding]

        # replicate target_emb along the second dimension
        target_emb_tiled = target_emb.repeat(1, H.size()[1], 1)  # [batch_size, Seq_len, embedding]

        # concatenate x and z along the last dimension
        e = torch.cat([x_emb, target_emb_tiled], dim=2)  # shape: (batch_size, Seq_len, embedding * 2)
        M = torch.tanh(e)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # shape: (batch_size, Seq_len, 1)

        out = H * alpha  # shape: (batch_size, Seq_len, 256)
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
