# Muti-head Attention 机制的实现
from math import sqrt
import torch
import torch.nn


class Conversation_Self_Attention(nn.Module):
    # input : batch_size * seq_len(conversation_size) * input_dim(sent_emb 512)
    # q : batch_size * input_dim(sent_emb 512) * dim_k(64)
    # k : batch_size * input_dim(sent_emb 512) * dim_k(64)
    # v : batch_size * input_dim(sent_emb 512) * dim_v(64)
    def __init__(self, input_dim, dim_k, dim_v):
        super(Conversation_Self_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)

        self._norm_fact = 1 / sqrt(2 * input_dim)

        self.pos_q = nn.Linear(input_dim, input_dim)
        self.pos_k = nn.Linear(input_dim, input_dim)

        self.bias = nn.Parameter(torch.Tensor(input_dim))
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        text_emb = x['sent_emb'] #batchsize * conversation_size * input_dim
        pos_emb = x['pos_emb'] # batchsize * input_dim * input_dim
        branch_emb = x['branch_emb'] #batchsize * conversation_size?

        #seq_len ---> (conversation_size)
        #input_dim ---> (sent_emb_size)
        Q = self.q(text_emb)  # Q: batch_size * seq_len * dim_k
        K = self.k(text_emb)  # K: batch_size * seq_len * dim_k
        V = self.v(text_emb)

        text_score = torch.bmm(Q, K.permute(0, 2, 1)) * self._norm_fact  # Q * K.T() # batch_size * seq_len * input_dim(sent_emb 512)

        Uq = self.pos_q(pos_emb)
        Uk = self.pos_k(pos_emb)

        pos_score = torch.bmm(Uq, Uk.permute(0, 2, 1)) * self._norm_fact  # Q * K.T() # batch_size * input_dim * input_dim

        bias = self.bias #????????

        att = text_score + pos_score + bias
        att_softmax = self.softmax(u) # 4.Softmax
        output = torch.bmm(att_softmax, V.permute(0, 2, 1)) # 5.Output
        return att_softmax, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, input_dim, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = Conversation_Self_Attention()

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        attn, output = self.attention(self.input_dim, k, v) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output
