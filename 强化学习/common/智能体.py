import os

import numpy as np
import torch
import torch as T
from torch.distributions import Categorical
from torch.nn import functional as F

from common import get_model, create_masks, 打印抽样数据
from common.PPO_数据集 import PPO_数据集
from common.TransformerConfig import TransformerConfig


class 智能体:
    def __init__(self, 动作数, 输入维度, 优势估计参数G=0.9999, 学习率=0.0003, 泛化优势估计参数L=0.985,
                 策略裁剪幅度=0.2, 并行条目数=64, 轮数=10, 熵系数=0.01):
        self.优势估计参数G = 优势估计参数G
        self.策略裁剪幅度 = 策略裁剪幅度
        self.轮数 = 轮数
        self.熵系数 = 熵系数
        self.泛化优势估计参数L = 泛化优势估计参数L
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        from common.env import 模型名称
        config = TransformerConfig()
        # print(f"模型名称: {模型名称}")
        model = get_model(config, 130, 模型名称)
        model = model.cuda(device)
        self.动作 = model
        self.优化函数 = torch.optim.Adam(self.动作.parameters(), lr=2e-5, betas=(0.9, 0.95), eps=1e-9)
        self.数据集 = PPO_数据集(并行条目数)
        self.文件名集 = []

    def 记录数据(self, 状态, 动作, 动作概率, 评价, 回报, 完结, 计数):
        self.数据集.记录数据(状态, 动作, 动作概率, 评价, 回报, 完结, 计数)

    def 存硬盘(self, 文件名):
        self.数据集.存硬盘(文件名)
        self.文件名集.append(文件名)

    def 读硬盘(self, 文件名):
        self.数据集.读硬盘(文件名)

    def 保存模型(self, 轮号):
        # print(f'... 保存模型 {轮号}...')
        from common.env import 保存模型名称
        torch.save(self.动作.state_dict(), f'E:/ai-play-wzry/weights/{保存模型名称}')
        if not os.path.isdir(f'E:/ai-play-wzry/weights/temp'):
            os.makedirs(f'E:/ai-play-wzry/weights/temp')
        torch.save(self.动作.state_dict(),
                   f'E:/ai-play-wzry/weights/temp/{保存模型名称.replace(".pth", "")}_{轮号}.pth')

    def 载入模型(self):
        # print('... 载入模型 ...')
        self.动作.载入权重()

    def 选择动作(self, 状态, device, 传入动作, 手动=False):
        self.动作.requires_grad_(False)
        操作序列 = torch.from_numpy(状态['操作序列'].astype(np.int64)).cuda(device)
        图片张量 = torch.from_numpy(状态['图片张量']).cuda(device)
        trg_mask = 状态['trg_mask']
        分布, 价值 = self.动作(图片张量, 操作序列, trg_mask)
        价值 = 价值[:, - 1, :]
        分布 = F.softmax(分布, dim=-1)
        分布 = 分布[:, - 1, :]
        分布 = Categorical(分布)
        if 手动:
            动作 = 传入动作
        else:
            动作 = 分布.sample()
        动作概率 = T.squeeze(分布.log_prob(动作)).item()
        动作 = T.squeeze(动作).item()
        return 动作, 动作概率, 价值

    def 选择动作批量(self, 状态, device, 目标输出_分_torch, 手动=False):
        # 分布,q_ = self.动作(状态)
        # r_, 价值 = self.评论(状态)
        self.动作.requires_grad_(False)
        操作序列 = torch.from_numpy(状态['操作序列'].astype(np.int64)).cuda(device)
        图片张量 = torch.from_numpy(状态['图片张量']).cuda(device)
        trg_mask = 状态['trg_mask']
        分布, 价值 = self.动作(图片张量, 操作序列, trg_mask)
        分布 = F.softmax(分布, dim=-1)
        分布 = Categorical(分布)
        if 手动:
            动作 = 目标输出_分_torch
        else:
            动作 = 分布.sample()
        动作概率 = T.squeeze(分布.log_prob(动作))
        动作 = T.squeeze(动作)
        return 动作, 动作概率, 价值

    def 学习(self, device):
        for i in range(1):
            总损失 = None
            for _ in range(self.轮数):
                动作集, 旧_动作概率集, 评价集, 回报集, 完结集, 图片集合, 动作数组, 条目集 = self.数据集.提取数据()
                # print('回报集', 回报集[0:10])
                价值 = 评价集

                优势函数值 = np.zeros(len(回报集), dtype=np.float32)

                for t in range(len(回报集) - 1):
                    折扣率 = 1
                    优势值 = 0
                    折扣率 = self.优势估计参数G * self.泛化优势估计参数L
                    计数 = 0
                    for k in range(t, len(回报集) - 1):

                        优势值 += pow(折扣率, abs(0 - 计数)) * (
                                回报集[k] + self.优势估计参数G * 价值[k + 1] * (1 - int(完结集[k])) - 价值[k])
                        计数 = 计数 + 1
                        if (1 - int(完结集[k])) == 0 or 计数 > 100:
                            break
                    优势函数值[t] = 优势值
                    # https://blog.csdn.net/zhkmxx930xperia/article/details/88257891
                    # GAE的形式为多个价值估计的加权平均数
                优势函数值 = T.tensor(优势函数值).to(device)

                价值 = T.tensor(价值).to(device)
                for 条 in 条目集:
                    条末 = 条[-1:]

                    旧_动作概率s = T.tensor(旧_动作概率集[条末]).to(device)
                    动作s = T.tensor(动作集[条末]).to(device)

                    self.动作.requires_grad_(True)

                    操作序列 = torch.from_numpy(动作数组[条].astype(np.int64)).cuda(device)
                    图片张量 = torch.from_numpy(图片集合[:, 条, :]).cuda(device).float()
                    src_mask, trg_mask = create_masks(操作序列.unsqueeze(0), 操作序列.unsqueeze(0), device)
                    分布, 评价结果 = self.动作(图片张量, 操作序列, trg_mask)
                    分布 = 分布[:, -1:, :]
                    评价结果 = 评价结果[:, -1:, :]

                    分布 = F.softmax(分布, dim=-1)
                    # 分布 = 分布[:, - 1, :]
                    # 评价结果 = 评价结果[:, - 1, :]
                    评价结果 = T.squeeze(评价结果)
                    分布 = Categorical(分布)
                    熵损失 = torch.mean(分布.entropy())
                    新_动作概率s = 分布.log_prob(动作s)
                    # 概率比 = 新_动作概率s.exp() / 旧_动作概率s.exp()
                    # # prob_ratio = (new_probs - old_probs).exp()
                    # 加权概率 = 优势函数值[条末] * 概率比
                    # 加权_裁剪_概率 = T.clamp(概率比, 1 - self.策略裁剪幅度,
                    #                                  1 + self.策略裁剪幅度) * 优势函数值[条末]
                    # 动作损失 = -T.min(加权概率, 加权_裁剪_概率).mean()

                    总回报 = 优势函数值[条末] + 价值[条末]
                    动作损失 = -总回报 * 新_动作概率s
                    动作损失 = 动作损失.mean()
                    评价损失 = (总回报 - 评价结果) ** 2
                    评价损失 = 评价损失.mean()

                    总损失 = 动作损失 + 0.5 * 评价损失 - self.熵系数 * 熵损失
                    # print(总损失)

                    self.优化函数.zero_grad()
                    # self.优化函数_评论.zero_grad()
                    总损失.backward()
                    self.优化函数.step()
                # self.优化函数_评论.step()
                # print('总损失', 总损失)

        self.数据集.清除数据()
        self.文件名集 = []

    def 监督强化学习(self, device, 状态, 回报, 动作, 动作可能性, 评价):
        # print(device,状态,回报,动作,动作可能性,评价)
        # for k, v in self.动作.named_parameters():
        #
        #     if k == '评价.weight' or k=='评价.bias':
        #         v.requires_grad = True
        回报集 = 回报
        价值 = 评价.cpu().numpy()[0, :, 0]
        优势函数值 = np.zeros(回报集.shape[0], dtype=np.float32)
        for t in range(len(回报集) - 1):
            折扣率 = 1
            优势值 = 0
            折扣率 = self.优势估计参数G * self.泛化优势估计参数L
            计数 = 0
            for k in range(t, len(回报集) - 1):

                优势值 += pow(折扣率, abs(0 - 计数)) * (回报集[k])
                计数 = 计数 + 1
                if 计数 > 200:
                    break
            优势函数值[t] = 优势值

            价值 = T.tensor(价值).to(device)
        for i in range(3):
            优势函数值 = T.tensor(优势函数值).to(device)
            旧_动作概率s = T.tensor(动作可能性).to(device)
            动作s = T.tensor(动作).to(device)

            self.动作.requires_grad_(True)

            操作序列 = torch.from_numpy(状态['操作序列'].astype(np.int64)).cuda(device)
            图片张量 = torch.from_numpy(状态['图片张量']).cuda(device).float()
            trg_mask = 状态['trg_mask']

            分布, 评价结果 = self.动作(图片张量, 操作序列, trg_mask)

            分布 = F.softmax(分布, dim=-1)
            # 分布 = 分布[:, - 1, :]
            # 评价结果 = 评价结果[:, - 1, :]
            评价结果 = T.squeeze(评价结果)
            分布 = Categorical(分布)
            # 熵损失 = torch.mean(分布.entropy())
            新_动作概率s = 分布.log_prob(动作s)
            # 旧_动作概率s=旧_动作概率s.exp()
            # 概率比 = 新_动作概率s / 旧_动作概率s
            # # prob_ratio = (new_probs - old_probs).exp()
            # 加权概率 = 优势函数值 * 概率比
            # 加权_裁剪_概率 = T.clamp(概率比, 1 - self.策略裁剪幅度,
            #                    1 + self.策略裁剪幅度) * 优势函数值
            # 动作损失 = -T.min(加权概率, 加权_裁剪_概率).mean()
            # 概率比2 = 新_动作概率s.mean() / 旧_动作概率s.mean()
            总回报 = 优势函数值  # + 价值
            动作损失 = -总回报 * 新_动作概率s
            动作损失 = 动作损失.mean()
            # 评价损失 = (总回报 - 评价结果) ** 2
            # 评价损失 = 评价损失.mean()
            # print(总回报[10:20], 新_动作概率s[:, 10:20].exp())

            总损失 = 动作损失  # + 0.5 * 评价损失 - self.熵系数 * 熵损失
            # print(总损失)

            self.优化函数.zero_grad()
            # self.优化函数_评论.zero_grad()
            总损失.backward()
            self.优化函数.step()
        # self.优化函数_评论.step()

    def 监督强化学习A(self, device, 状态, 回报, 动作, 动作可能性, 评价, 完结集):
        # print(device,状态,回报,动作,动作可能性,评价)
        # for k, v in self.动作.named_parameters():
        #
        #     if k == '评价.weight' or k=='评价.bias':
        #         v.requires_grad = True
        回报集 = 回报
        价值 = 评价.cpu().numpy()[0, :, 0]
        优势函数值 = np.zeros(回报集.shape[0], dtype=np.float32)
        for t in range(len(回报集) - 1):
            折扣率 = 1
            优势值 = 0
            折扣率 = self.优势估计参数G * self.泛化优势估计参数L
            计数 = 0
            for k in range(t, len(回报集) - 1):

                优势值 += pow(折扣率, abs(0 - 计数)) * (回报集[k] * (1 - 完结集[0, k] * 0))
                计数 = 计数 + 1
                if 计数 > 200 or 完结集[0, k] == 2111111:
                    break
            优势函数值[t] = 优势值

            价值 = T.tensor(价值).to(device)
        for i in range(3):
            优势函数值 = T.tensor(优势函数值).to(device)
            旧_动作概率s = T.tensor(动作可能性).to(device)
            动作s = T.tensor(动作).to(device)

            self.动作.requires_grad_(True)

            操作序列 = torch.from_numpy(状态['操作序列'].astype(np.int64)).cuda(device)
            图片张量 = torch.from_numpy(状态['图片张量']).cuda(device).float()
            trg_mask = 状态['trg_mask']

            分布, 评价结果 = self.动作(图片张量, 操作序列, trg_mask)

            分布 = F.softmax(分布, dim=-1)
            # 分布 = 分布[:, - 1, :]
            # 评价结果 = 评价结果[:, - 1, :]
            评价结果 = T.squeeze(评价结果)
            分布 = Categorical(分布)
            # 熵损失 = torch.mean(分布.entropy())
            新_动作概率s = 分布.log_prob(动作s)
            # 旧_动作概率s=旧_动作概率s.exp()
            # 概率比 = 新_动作概率s / 旧_动作概率s
            # # prob_ratio = (new_probs - old_probs).exp()
            # 加权概率 = 优势函数值 * 概率比
            # 加权_裁剪_概率 = T.clamp(概率比, 1 - self.策略裁剪幅度,
            #                    1 + self.策略裁剪幅度) * 优势函数值
            # 动作损失 = -T.min(加权概率, 加权_裁剪_概率).mean()
            # 概率比2 = 新_动作概率s.mean() / 旧_动作概率s.mean()
            总回报 = 优势函数值  # + 价值
            动作损失 = -总回报 * 新_动作概率s
            动作损失 = 动作损失.mean()
            # 评价损失 = (总回报 - 评价结果) ** 2
            # 评价损失 = 评价损失.mean()
            # print(总回报[10:20], 新_动作概率s[:, 10:20].exp())

            总损失 = 动作损失  # + 0.5 * 评价损失 - self.熵系数 * 熵损失
            # print(总损失)

            self.优化函数.zero_grad()
            # self.优化函数_评论.zero_grad()
            总损失.backward()
            self.优化函数.step()
        # self.优化函数_评论.step()

    def 监督学习(self, 状态, 目标输出, 打印, 数_词表, 操作_分_torch, device):
        分布, 价值 = self.动作(状态, device)
        lin = 分布.view(-1, 分布.size(-1))
        _, 抽样 = torch.topk(分布, k=1, dim=-1)
        抽样np = 抽样.cpu().numpy()

        self.优化函数.zero_grad()
        loss = F.cross_entropy(lin, 目标输出.contiguous().view(-1), ignore_index=-1)
        if 打印:
            # print(loss)
            打印抽样数据(数_词表, 抽样np[0:1, :, :], 操作_分_torch[0, :])
        loss.backward()

        self.优化函数.step()

    def 选择动作_old(self, 状态):

        # 分布,q_ = self.动作(状态)
        # r_, 价值 = self.评论(状态)
        输出_实际_A, 价值 = self.动作(状态)

        输出_实际_A = F.softmax(输出_实际_A, dim=-1)
        输出_实际_A = 输出_实际_A[:, - 1, :]
        抽样 = torch.multinomial(输出_实际_A, num_samples=1)
        抽样np = 抽样.cpu().numpy()
        return 抽样np[0, -1]
