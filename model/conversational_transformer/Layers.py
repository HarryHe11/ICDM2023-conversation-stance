import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, u_qs, u_ks, thetas, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, u_qs, u_ks, thetas, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)


    def forward(self, enc_input, enc_abs_pos, theta_indexs, slf_attn_mask=None):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input, enc_abs_pos, enc_abs_pos, theta_indexs, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, u_qs, u_ks, thetas, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.u_qs = u_qs
        self.u_ks = u_ks
        self.thetas = thetas

        self.attention = ScaledDotProductAttention(temperature=2 * d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, q_u, k_u, theta_indexes, mask=None):
        # v: b x len x d_model
        # q_u: b x len x d_model
        # theta_indexes: b x len x len x 2
        
        d_model, d_k, d_v, n_head = self.d_model, self.d_k, self.d_v, self.n_head
        sz_b, length = q.size(0), q.size(1)

        residual = q

        # Pass through the pre-attention projection: b x len x (n * dv)
        # Separate different heads: b x len x n x dv
        q = self.w_qs(q).view(sz_b, length, n_head, d_k)
        k = self.w_ks(k).view(sz_b, length, n_head, d_k)
        v = self.w_vs(v).view(sz_b, length, n_head, d_v)
        q_u = self.u_qs(q_u).view(sz_b, length, n_head, d_model)
        k_u = self.u_ks(k_u).view(sz_b, length, n_head, d_model)

        # Transpose for attention dot product: b x n x len x dv
        q, k, v, q_u, k_u = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), q_u.transpose(1, 2), k_u.transpose(1, 2)
        
        # construct theta: b x n x len x len
        theta = torch.cat(sz_b * length * length * [self.thetas]).view(sz_b, length, length, n_head, 2).transpose(1, 3)
        theta_indexes = torch.cat(n_head * [theta_indexes.view(1, sz_b, length, length, 2)]).transpose(0, 1)
        theta = torch.sum(torch.mul(theta, theta_indexes), -1)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = self.attention(q, k, v, q_u, k_u, theta, mask=mask)

        # Transpose to move the head dimension back: b x len x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x len x (n * dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, length, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
    

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, q_u, k_u, theta, mask=None):
        # q: b x n x len x dk
        # k: b x n x len x dk
        # v: b x n x len x dv
        # q_u: b x n x len x d_model
        # k_u: b x n x len x d_model
        # theta: b x n x len x len
        
        term1 = torch.matmul(q / self.temperature, k.transpose(2, 3))
        term2 = torch.matmul(q_u / self.temperature, k_u.transpose(2, 3))
        attn = term1 + term2 + theta

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output