# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'branch_bert'
        self.train_path = dataset + '/data/branch_train.txt'                                # 训练集
        self.dev_path = dataset + '/data/branch_dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/branch_test.txt'                                  # 测试集
        self.class_list = [0, 1]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 500                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        self.bert_path = 'bert_pretrain/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.weights = [1.0, 1.0]

class Model(nn.Module):
    '''Branch-BERT w/o CFE'''
    def __init__(self, config, mode='branch'):
        super(Model, self).__init__()
        self.config = config
        self.mode = mode
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.droptout = nn.Dropout(0.1)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        seq_len = x[1]
        mask = x[2]
        cut_idx = x[3]
        output = self.bert(context, attention_mask=mask)
        last_hidden_state = output.last_hidden_state #(batch_size,max_lenth, hidden)
        ts_list = []
        for i in range(cut_idx.size()[0]):
            target_tensor = last_hidden_state[i ,cut_idx[i]:seq_len[i], : ].unsqueeze(0)
            assert target_tensor.size() == torch.Size([1, seq_len[i]-cut_idx[i], 768])
            output_tensor = torch.mean(target_tensor, dim=1) #(1, hidden)
            ts_list.append(output_tensor)
        pooler_output= torch.cat(ts_list, dim=0)# ([batch_size, hidden_size])
        out = self.droptout(pooler_output)
        out = self.fc(out)
        return out