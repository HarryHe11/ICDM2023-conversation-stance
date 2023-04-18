import math
import torch
import torch.nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, time_diff_tensor, max_len=100):
        '''
            d_model: sentence embedding size
            time_diff_tensor: Tensor([s(i) - s(1) for i in range(len(conversation_size))])
            max_len: max size of a conversation
        '''

        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = time_diff_tensor.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]