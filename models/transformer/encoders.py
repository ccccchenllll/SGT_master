from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import enMultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, memory,d_model=512, d_k=64, d_v=64, h=8, d_ff=512, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = enMultiHeadAttention(memory,d_model, d_k, d_v, h, dropout,
                                        identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values,matrix, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, matrix,attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, memory,d_model=512, d_k=64, d_v=64, h=8, d_ff=512, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.memory = memory
        self.layers = nn.ModuleList([EncoderLayer(memory,d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                                  #N = 3
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input,matrix, attention_weights=None):
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        outs = []
        out = input
        ad_matrix = matrix
        for l in self.layers:
            out = l(out, out, out, ad_matrix,attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))


        outs = torch.cat(outs, 1)

        return outs, attention_mask


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx,memory, d_in=512, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, memory, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.ma_fc = nn.Linear(10,10)
        self.ma_norm = nn.LayerNorm(10)


    def forward(self, input, matrix, attention_weights=None):
        out = F.relu(self.fc(input))  #(bactchsize,6,512)
        out = self.dropout(out)
        out = self.layer_norm(out)  #(10,6,512)

        out_matrix = F.relu((self.ma_fc(matrix)))
        out_matrix = self.dropout(out_matrix)

        return super(MemoryAugmentedEncoder, self).forward(out,out_matrix, attention_weights=attention_weights)
