import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = Embedder2(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # from torch.autograd import Variable
    # def forward(self, x):
    #     # make embeddings relatively larger
    #     x = x * math.sqrt(self.d_model)
    #     # add constant to embedding
    #     seq_len = x.size(1)
    #     pe = Variable(self.pe[:, :seq_len], requires_grad=False)
    #     if x.is_cuda:
    #         pe.cuda()
    #     x = x + pe
    #     x = self.dropout(x)
    #     return x
    def forward(self, x):
        """在较新的 PyTorch 版本中，不需要使用 Variable，直接使用张量操作。如果要防止某些张量被计算梯度，可以使用 .detach() 方法。"""
        # make embeddings relatively larger
        x *= math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].detach()
        if x.is_cuda:
            pe = pe.cuda(x.device)
        x = x + pe
        x = self.dropout(x)
        return x


class Embedder2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedder2, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            np.random.seed(1)
            np数 = np.random.uniform(0, 1, (num_embeddings, embedding_dim))
            self.weight = nn.Parameter(torch.Tensor(np数))
            # self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            # self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, item):
        return F.embedding(
            item, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
