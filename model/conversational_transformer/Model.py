import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformer.Layers import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):

        super().__init__()

        self.u_qs = nn.Linear(d_model, n_head * d_model, bias=False) # shared across layers
        self.u_ks = nn.Linear(d_model, n_head * d_model, bias=False) # shared across layers
        self.thetas = nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(1, n_head, 2), gain=1.0), requires_grad=True)
        
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, self.u_qs, self.u_ks, self.thetas, dropout=dropout) for _ in range(n_layers)])
        
        self.output_layer = nn.Linear(d_model, 3, bias=True)

    def forward(self, enc_input, enc_abs_pos, theta_indexs, mask):
        # enc_input: b x len x d_model
        # enc_abs_pos: b x len x d_model
        # theta_indexs: b x len x len x 2
        # mask: b x len x len
        
        for enc_layer in self.layer_stack:
            enc_input = enc_layer(enc_input, enc_abs_pos, theta_indexs, slf_attn_mask=mask)

        output = F.softmax(self.output_layer(enc_input), dim=-1)
        return output