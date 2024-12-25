import numpy as np

from common import save_obj, load_obj


class PPO_数据集:
    def __init__(self, 并行条目数量):
        # self.状态集 = []
        self.动作概率集 = []
        self.评价集 = []
        self.动作集 = []
        self.回报集 = []
        self.完结集 = []

        self.并行条目数量 = 并行条目数量
        self.完整数据 = {}
        self.图片信息 = np.ones([1, 1000, 6 * 6 * 2048], dtype='float')
        self.操作信息 = np.ones((0,))

    def 提取数据(self):
        状态集_长度 = len(self.回报集)
        条目_起始位 = np.arange(0, 状态集_长度 - 100, self.并行条目数量)
        下标集 = np.arange(状态集_长度, dtype=np.int64)

        条目集 = [下标集[i:i + self.并行条目数量] for i in 条目_起始位]

        return np.array(self.动作集), \
            np.array(self.动作概率集), \
            self.评价集, \
            np.array(self.回报集), \
            np.array(self.完结集), \
            self.图片信息, \
            self.操作信息, \
            条目集

    def 记录数据(self, 状态, 动作, 动作概率, 评价, 回报, 完结, 计数):
        # self.状态集.append(状态)
        self.动作集.append(动作)
        self.动作概率集.append(动作概率)
        self.评价集.append(评价)
        self.回报集.append(回报)
        self.完结集.append(完结)
        self.图片信息[:, 计数, :] = 状态['图片张量']
        self.操作信息 = np.append(self.操作信息, 状态['操作序列'])

    def 清除数据(self):
        self.图片信息 = []
        self.动作概率集 = []
        self.动作集 = []
        self.回报集 = []
        self.完结集 = []
        self.评价集 = []
        self.完整数据 = {}
        # del self.状态集,self.动作概率集,self.评价集,self.动作集,self.回报集,self.完结集,self.完整数据
        # gc.collect()

    def 存硬盘(self, 文件名):
        self.完整数据['图片信息'] = self.图片信息[:, 0:len(self.动作集), :]
        self.完整数据['动作概率集'] = self.动作概率集
        self.完整数据['动作集'] = self.动作集
        self.完整数据['回报集'] = self.回报集
        self.完整数据['完结集'] = self.完结集
        self.完整数据['评价集'] = self.评价集
        self.完整数据['操作信息'] = self.操作信息
        save_obj(self.完整数据, 文件名)
        self.完整数据 = {}
        # self.图片信息 = []
        self.动作概率集 = []
        self.动作集 = []
        self.回报集 = []
        self.完结集 = []
        self.评价集 = []
        # self.操作信息=[]

        # del self.图片信息,self.动作概率集,self.评价集,self.动作集,self.回报集,self.完结集,self.完整数据
        # gc.collect()

    def 读硬盘(self, 文件名):
        self.完整数据 = load_obj(文件名)
        self.图片信息 = self.完整数据['图片信息']
        self.动作概率集 = self.完整数据['动作概率集']
        self.动作集 = self.完整数据['动作集']
        self.回报集 = self.完整数据['回报集']
        self.完结集 = self.完整数据['完结集']
        self.评价集 = self.完整数据['评价集']
        self.操作信息 = self.完整数据['操作信息']
        self.完整数据 = {}
