import math
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

import json
from common.Batch import create_masks
from common.Sublayers import 全连接层
from common.env import 状态词典B, 判断状态模型地址
from common.resnet_utils import myResnet
from common.模型_策略梯度 import Transformer

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101 = torchvision.models.resnet101(pretrained=True).eval()
resnet101 = myResnet(resnet101).cuda(device).requires_grad_(False)
from random import shuffle


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class 判断状态(nn.Module):
    def __init__(self, 种类数, 隐藏层尺寸, 输入层尺寸=2048, 输入尺寸A=36):
        super().__init__()
        self.隐藏层尺寸 = 隐藏层尺寸
        self.输入层尺寸 = 输入层尺寸
        self.输入尺寸A = 输入尺寸A
        self.输入层 = 全连接层(输入层尺寸, 隐藏层尺寸)
        self.隐藏层 = 全连接层(隐藏层尺寸, 隐藏层尺寸)
        self.输出层 = 全连接层(隐藏层尺寸 * 输入尺寸A, 种类数)

    def forward(self, 图向量):
        图向量 = 图向量.reshape((图向量.shape[0], self.输入尺寸A, self.输入层尺寸))
        中间量 = gelu(self.输入层(图向量))
        中间量 = self.隐藏层(中间量)
        中间量 = 中间量.reshape((中间量.shape[0], self.隐藏层尺寸 * self.输入尺寸A))
        结果 = self.输出层(中间量)
        return 结果

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

model_判断状态 = Transformer(6, 768, 2, 12, 0.0, 6 * 6 * 2048).cuda(device)
optimizer = torch.optim.Adam(model_判断状态.parameters(), lr=6.25e-5, betas=(0.9, 0.98), eps=1e-9)
路径json = r'E:\ai-play-wzry\判断数据样本\判断新.json'
全部数据 = {}
状态列表 = []
for K in 状态词典B:
    状态列表.append(K)
with open(路径json, encoding='utf-8') as f:
    while True:
        df = f.readline()
        df = df.replace('\'', '\"')
        if df == "":
            break
        单元 = json.loads(df)
        for key in 单元:
            全部数据[key] = 单元[key]

状态 = np.ones((1,), dtype='int64')
for i in range(100):
    打乱顺序 = random_dic(全部数据)
    for key in 打乱顺序:
        状态编号 = 状态词典B[全部数据[key]]
        状态[0] = 状态编号
        目标输出 = torch.from_numpy(状态).cuda(device)
        图片路径 = r'E:\ai-play-wzry\判断数据样本\\' + key + '.jpg'
        if os.path.isfile(图片路径):
            img = Image.open(图片路径)
        else:
            img = Image.open(
                r'E:\ai-play-wzry\训练数据样本\未用\\' + key.split('_')[0] + '\\' + key.split('_')[1] + '.jpg')
        img2 = np.array(img)
        img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1).float() / 255
        _, out = resnet101(img2)
        图片张量 = out.reshape(1, 6 * 6 * 2048)
        操作序列 = np.ones((1, 1))
        操作张量 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(操作张量.unsqueeze(0), 操作张量.unsqueeze(0), device)
        实际输出, _ = model_判断状态(图片张量.unsqueeze(0), 操作张量.unsqueeze(0), trg_mask)
        _, 抽样 = torch.topk(实际输出, k=1, dim=-1)
        抽样np = 抽样.cpu().numpy()
        optimizer.zero_grad()
        实际输出 = 实际输出.view(-1, 实际输出.size(-1))
        loss = F.cross_entropy(实际输出, 目标输出.contiguous().view(-1), ignore_index=-1)
        loss.backward()
        optimizer.step()
        print('轮', i + 1, '实际输出', 状态列表[抽样np[0, 0, 0, 0]], '目标输出', 全部数据[key], loss)
        if (i + 1) % 25 == 0:
            torch.save(model_判断状态.state_dict(),
                       f"E:/ai-play-wzry/weights/temp/model_weights_judgment_state_{i + 1}.pth")
    torch.save(model_判断状态.state_dict(), 判断状态模型地址)
