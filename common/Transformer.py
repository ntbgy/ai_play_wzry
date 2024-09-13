from torch import nn as nn

from common.Decoder import Decoder
from common.Sublayers import 全连接层


class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout, 图向量尺寸=6 * 6 * 2048):
        super().__init__()
        self.图转 = 全连接层(图向量尺寸, d_model)

        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.outX = 全连接层(d_model, trg_vocab)

        self.评价 = 全连接层(d_model, 1)

    def forward(self, 图向量, 操作, trg_mask):
        图向量 = self.图转(图向量)

        d_output = self.decoder(图向量, 操作, trg_mask)
        output = self.outX(d_output)
        评价 = self.评价(d_output)
        return output, 评价
