import copy

import torch
from torch import nn as nn

from common.Embed import Embedder
from common.Layers import DecoderLayer
from common.Sublayers import Norm


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, 最大长度=1024):
        super().__init__()
        self.N = N
        self.embedX = Embedder(vocab_size, d_model)
        self.embedP = Embedder(最大长度, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, 图向量, 操作, trg_mask):
        position = torch.arange(0, 图向量.size(1), dtype=torch.long,
                                device=图向量.device)

        x = 图向量 + self.embedP(position) + self.embedX(操作) * 0

        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
