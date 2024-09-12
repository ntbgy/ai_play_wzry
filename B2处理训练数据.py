import os

import numpy as np
import torch
import torchvision
from PIL import Image

import json
from common.resnet_utils import myResnet

操作记录 = 'E:/ai-play-wzry/训练数据样本/未用'
if not os.path.exists(操作记录):
    os.makedirs(操作记录)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101 = torchvision.models.resnet101(pretrained=True).eval()
resnet101 = myResnet(resnet101).cuda(device).requires_grad_(False)
词数词典路径 = "./json/词_数表.json"

with open(词数词典路径, encoding='utf8') as f:
    词数词典 = json.load(f)
dirnames = list()
for dirpath, dirnames, filenames in os.walk(操作记录):
    if len(dirnames) > 0:
        break
for 号 in dirnames:
    路径json = 操作记录 + '/' + 号 + '/_操作数据.json'
    numpy数组路径 = 操作记录 + '/' + 号 + '/图片_操作预处理数据2.npz'
    # 如果已经处理过了就不用重复处理了
    if os.path.isfile(numpy数组路径):
        continue

    图片张量 = torch.Tensor(0)

    操作张量 = torch.Tensor(0)

    伪词序列 = torch.from_numpy(np.ones((1, 60)).astype(np.int64)).cuda(device).unsqueeze(0)

    操作序列 = np.ones((1, 1))
    结束序列 = np.ones((1, 1))
    计数 = 0
    print('正在处理{}'.format(号))
    数据列 = []
    with open(路径json, encoding='utf-8') as f:
        移动操作 = '无移动'
        while True:
            df = f.readline()
            df = df.replace('\'', '\"')

            if df == "":
                break
            try:
                df = json.loads(df)
            except json.decoder.JSONDecodeError as e:
                print(df)
                break
            数据列.append(df)

    with open(路径json, encoding='utf-8') as f:
        移动操作 = '无移动'
        for i in range(len(数据列)):
            df = 数据列[i]

            if 图片张量.shape[0] == 0:
                img = Image.open(操作记录 + '/' + 号 + '/{}.jpg'.format(df["图片号"]))
                img2 = np.array(img)

                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _, out = resnet101(img2)
                图片张量 = out.reshape(1, 6 * 6 * 2048)
                移动操作a = df["移动操作"]
                if 移动操作a != '无移动':
                    移动操作 = 移动操作a

                操作序列[0, 0] = 词数词典[移动操作 + "_" + df["动作操作"]]
                结束序列[0, 0] = df["结束"]
            else:
                img = Image.open(操作记录 + '/' + 号 + '/{}.jpg'.format(df["图片号"]))
                img2 = np.array(img)

                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _, out = resnet101(img2)

                图片张量 = torch.cat((图片张量, out.reshape(1, 6 * 6 * 2048)), 0)
                移动操作a = df["移动操作"]
                if 移动操作a != '无移动':
                    移动操作 = 移动操作a
                操作序列 = np.append(操作序列, 词数词典[移动操作 + "_" + df["动作操作"]])
                结束序列 = np.append(结束序列, df["结束"])
            # 手动释放不再需要的 GPU 张量占用的内存
            del img
            del img2
            torch.cuda.empty_cache()

        图片张量np = 图片张量.cpu().numpy()
        操作序列 = 操作序列.astype(np.int64)
        np.savez(numpy数组路径, 图片张量np=图片张量np, 操作序列=操作序列, 结束序列=结束序列)
