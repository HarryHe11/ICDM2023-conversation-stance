import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.Layers import EncoderLayer


class Config(object):
    """Config for parameters"""

    def __init__(self, dataset_path):
        self.model_name = 'Transformer'
        self.dataset_path = dataset_path
        self.train_path = dataset_path + \
            '/data/train.parquet'                    # train set
        self.test_path = dataset_path + \
            '/data/test_emb_branch.parquet'                                  # test set

        # label list
        self.class_list = range(2)
        # number of labels                       # number of label classes
        self.num_classes = len(self.class_list)
        self.save_path = dataset_path + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        # devices
        self.weight = torch.FloatTensor([1.0,1.0]).to(self.device)

        # number of training epochs
        self.num_epochs = 3
        # patience for early stopping
        self.patience = 3
        self.batch_size = 64                                           # size of batch
        # max sequence size
        self.learning_rate = 1e-5                                       # learning rate


class Encoder(nn.Module):

    def __init__(self, n_layers = 6, n_head = 8, d_k =64, d_v = 64, d_model = 384, d_inner=2048, dropout=0.1):

        super().__init__()

        self.u_qs = nn.Linear(d_model, n_head * d_model, bias=False) # shared across layers
        self.u_ks = nn.Linear(d_model, n_head * d_model, bias=False) # shared across layers
        self.thetas = nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(1, n_head, 2), gain=1.0), requires_grad=True)
        
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, self.u_qs, self.u_ks, self.thetas, dropout=dropout) for _ in range(n_layers)])
        
        self.output_layer = nn.Linear(d_model, 2, bias=True)

    def forward(self, x):
        # enc_input: b x len x d_model
        # enc_abs_pos: b x len x d_model
        # theta_indexs: b x len x len x 2
        # mask: b x len x len
        enc_input = x[0]
        enc_abs_pos = x[1]
        theta_indexs = x[2]
        mask = x[3]

        for enc_layer in self.layer_stack:
            enc_input = enc_layer(enc_input, enc_abs_pos, theta_indexs, slf_attn_mask=mask)
        # input: b x len x d_model
        # output: b x len x 3
        output = F.softmax(self.output_layer(enc_input), dim=-1)
        return output


        #   len x len x 2
        #       2: [0, 1]: same branch, [1, 0]: different branch
        # mask: b x len x len
        #   len = 4
        #   real_len = 2
        #   4 x 4 = 16
        #      
        # [
        #     [1, 1, 0 (masked), 0 (masked)],
        #     [],
        #     [],
        #     []
        # ]
        
        # t(i): avg_sec between two tweet in a conversation

