import os

import numpy as np
import torch
import torchvision
from PIL import Image

import json
from common.resnet_utils import myResnet

# 定义操作记录的路径
操作记录 = 'E:/ai-play-wzry/训练数据样本/未用'
# 如果操作记录路径不存在，则创建该路径
if not os.path.exists(操作记录):
    os.makedirs(操作记录)

# 设置设备为 GPU，如果可用，否则使用 CPU
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# 加载预训练的 ResNet101 模型，并设置为评估模式
resnet101 = torchvision.models.resnet101(pretrained=True).eval()
# 将 ResNet101 模型转换为自定义的 myResnet 模型，并移动到 GPU 上，且不需要梯度更新
resnet101 = myResnet(resnet101).cuda(device).requires_grad_(False)
# 定义词数词典的路径
词数词典路径 = "./json/词_数表.json"

# 打开词数词典文件，加载词数词典
with open(词数词典路径, encoding='utf8') as f:
    词数词典 = json.load(f)

# 初始化目录名列表
dirnames = list()
# 遍历操作记录路径下的所有文件和目录
for dirpath, dirnames, filenames in os.walk(操作记录):
    # 如果目录名列表不为空，则停止遍历
    if len(dirnames) > 0:
        break

# 遍历目录名列表
for 号 in dirnames:
    # 定义操作数据的 JSON 文件路径
    路径json = 操作记录 + '/' + 号 + '/_操作数据.json'
    # 定义预处理数据的 NPZ 文件路径
    numpy数组路径 = 操作记录 + '/' + 号 + '/图片_操作预处理数据2.npz'
    # 如果 NPZ 文件已经存在，则跳过当前目录的处理
    if os.path.isfile(numpy数组路径):
        continue

    # 初始化图片张量和操作张量
    图片张量 = torch.Tensor(0)
    操作张量 = torch.Tensor(0)
    # 初始化伪词序列
    伪词序列 = torch.from_numpy(np.ones((1, 60)).astype(np.int64)).cuda(device).unsqueeze(0)
    # 初始化操作序列和结束序列
    操作序列 = np.ones((1, 1))
    结束序列 = np.ones((1, 1))
    # 初始化计数变量
    计数 = 0
    # 打印正在处理的目录号
    print('正在处理 {}'.format(号))
    # 初始化数据列列表
    数据列 = []
    # 打开操作数据的 JSON 文件
    with open(路径json, encoding='utf-8') as f:
        # 初始化移动操作为 '无移动'
        移动操作 = '无移动'
        # 读取文件中的每一行
        while True:
            df = f.readline()
            # 将单引号替换为双引号
            df = df.replace('\'', '\"')
            # 如果文件读取完毕，则退出循环
            if df == "":
                break
            # 尝试将读取的行解析为 JSON 对象
            try:
                df = json.loads(df)
            # 如果解析失败，则打印错误信息并退出循环
            except json.decoder.JSONDecodeError as e:
                print(df)
                break
            # 将解析后的 JSON 对象添加到数据列列表中
            数据列.append(df)

    # 再次打开操作数据的 JSON 文件
    with open(路径json, encoding='utf-8') as f:
        # 初始化移动操作为 '无移动'
        移动操作 = '无移动'
        # 遍历数据列列表中的每一个元素
        for i in range(len(数据列)):
            df = 数据列[i]
            # 如果图片张量的形状为空，则读取图片并进行预处理
            if 图片张量.shape[0] == 0:
                # 打开图片文件
                img = Image.open(操作记录 + '/' + 号 + '/{}.jpg'.format(df["图片号"]))
                # 将图片转换为 numpy 数组
                img2 = np.array(img)
                # 将 numpy 数组转换为 torch 张量，并移动到 GPU 上，调整维度并归一化
                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                # 通过 ResNet101 模型提取图片特征
                _, out = resnet101(img2)
                # 将图片特征张量重塑为 (1, 6 * 6 * 2048) 的形状
                图片张量 = out.reshape(1, 6 * 6 * 2048)
                # 获取移动操作
                移动操作a = df["移动操作"]
                # 如果移动操作不为 '无移动'，则更新移动操作
                if 移动操作a != '无移动':
                    移动操作 = 移动操作a
                # 将移动操作和动作操作组合成词，并获取其在词数词典中的索引
                操作序列[0, 0] = 词数词典[移动操作 + "_" + df["动作操作"]]
                # 获取结束标志
                结束序列[0, 0] = df["结束"]
            # 如果图片张量的形状不为空，则继续处理下一张图片
            else:
                # 打开图片文件
                img = Image.open(操作记录 + '/' + 号 + '/{}.jpg'.format(df["图片号"]))
                # 将图片转换为 numpy 数组
                img2 = np.array(img)
                # 将 numpy 数组转换为 torch 张量，并移动到 GPU 上，调整维度并归一化
                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                # 通过 ResNet101 模型提取图片特征
                _, out = resnet101(img2)
                # 将新的图片特征张量添加到图片张量中
                图片张量 = torch.cat((图片张量, out.reshape(1, 6 * 6 * 2048)), 0)
                # 获取移动操作
                移动操作a = df["移动操作"]
                # 如果移动操作不为 '无移动'，则更新移动操作
                if 移动操作a != '无移动':
                    移动操作 = 移动操作a
                # 将移动操作和动作操作组合成词，并获取其在词数词典中的索引，然后添加到操作序列中
                操作序列 = np.append(操作序列, 词数词典[移动操作 + "_" + df["动作操作"]])
                # 获取结束标志，并添加到结束序列中
                结束序列 = np.append(结束序列, df["结束"])
            # 手动释放不再需要的 GPU 张量占用的内存
            del img
            del img2
            torch.cuda.empty_cache()

        # 将图片张量转换为 numpy 数组，并移动到 CPU 上
        图片张量np = 图片张量.cpu().numpy()
        # 将操作序列转换为 int64 类型
        操作序列 = 操作序列.astype(np.int64)
        # 将图片张量、操作序列和结束序列保存到 NPZ 文件中
        np.savez(numpy数组路径, 图片张量np=图片张量np, 操作序列=操作序列, 结束序列=结束序列)
