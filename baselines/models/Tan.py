# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'TAN'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                    
        self.test_path = dataset + '/data/test.txt'                                 
        self.class_list = [0,1]                                                      
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        self.weights = torch.tensor([1.0, 1.0])
        self.dropout = 0.1                                             
        self.require_improvement = 1000                                 
        self.num_classes = len(self.class_list)                         
        self.n_vocab = 0                                                
        self.num_epochs = 20                                          
        self.batch_size = 128                                          
        self.pad_size = 32                                            
        self.learning_rate = 1e-4                                   
        self.embed = 300           
        self.hidden_size = 256                                        
        self.num_layers = 2                                        
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
