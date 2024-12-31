from common import *
from common import 读出引索, Transformer
from common.Batch import create_masks
from common.TransformerConfig import TransformerConfig
from common.env import 状态词典, 状态词典B, training_data_save_directory
from common.total_seconds import total_seconds
from common.智能体 import 智能体

状态列表 = [K for K in 状态词典B]
训练数据保存目录 = 'E:/ai-play-wzry/训练数据样本/未用'
if not os.path.exists(训练数据保存目录):
    os.makedirs(训练数据保存目录)
dirs = list()
for root, dirs, files in os.walk('E:/ai-play-wzry/训练数据样本/未用'):
    if len(dirs) > 0:
        break
词数词典路径 = "./json/词_数表.json"
数_词表路径 = "./json/数_词表.json"
if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
    词_数表, 数_词表 = 读出引索(词数词典路径, 数_词表路径)
with open(词数词典路径, encoding='utf8') as f:
    词数词典 = json.load(f)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
config = TransformerConfig()
model_判断状态 = Transformer(6, 768, 2, 12, 0.0, 6 * 6 * 2048)
model_判断状态.load_state_dict(torch.load('E:/ai-play-wzry/weights/model_weights_judgment_state.pth'))
model_判断状态.cuda(device).requires_grad_(False)
N = 15000  # 运行N次后学习
条数 = 100
轮数 = 3
学习率 = 0.0003
智能体 = 智能体(
    动作数=7,
    并行条目数=条数,
    学习率=学习率,
    轮数=轮数,
    输入维度=6)
分块大小 = 600
游标大小 = 600
树枝 = 1
计数 = 0
time_start = time.time()
for j in range(100):
    for 号 in dirs:
        预处理数据 = os.path.join(training_data_save_directory, 号, '图片_操作预处理数据2.npz')
        if os.path.isfile(预处理数据):
            npz文件 = np.load(预处理数据, allow_pickle=True)
            图片张量np, 操作序列 = npz文件["图片张量np"], npz文件["操作序列"]
            if 图片张量np.shape[0] < 600:
                continue
            循环 = True
            游标 = 0
            操作序列 = np.insert(操作序列, 0, 128)
            操作_分_表 = []
            目标输出_分_表 = []
            图片_分_表 = []
            while 循环:
                if 游标 + 分块大小 < 操作序列.shape[0]:
                    操作_分 = 操作序列[游标:游标 + 分块大小]
                    目标输出_分 = 操作序列[游标 + 1:游标 + 1 + 分块大小]
                    图片_分 = 图片张量np[游标:游标 + 分块大小, :]
                    操作_分_表.append(操作_分)
                    目标输出_分_表.append(目标输出_分)
                    图片_分_表.append(图片_分)
                    游标 += 游标大小
                else:
                    操作_分 = 操作序列[-分块大小 - 1:-1]
                    目标输出_分 = 操作序列[-分块大小:]
                    图片_分 = 图片张量np[-分块大小:, :]
                    操作_分_表.append(操作_分)
                    目标输出_分_表.append(目标输出_分)
                    图片_分_表.append(图片_分)
                    循环 = False
            循环 = True
            i = 0
            while 循环:
                if (i + 1) * 树枝 < len(操作_分_表):
                    操作_分_枝 = np.array(操作_分_表[i * 树枝:(i + 1) * 树枝])
                    图片_分_枝 = np.array(图片_分_表[i * 树枝:(i + 1) * 树枝])
                    目标输出_分_枝 = np.array(目标输出_分_表[i * 树枝:(i + 1) * 树枝])
                else:
                    操作_分_枝 = np.array(操作_分_表[i * 树枝:len(操作_分_表)])
                    图片_分_枝 = np.array(图片_分_表[i * 树枝:len(图片_分_表)], dtype=np.float32)
                    目标输出_分_枝 = np.array(目标输出_分_表[i * 树枝:len(目标输出_分_表)])
                    循环 = False
                操作_分_torch = torch.from_numpy(操作_分_枝).cuda(device)
                操作序列A = np.ones_like(操作_分_枝)
                操作序列A_torch = torch.from_numpy(操作序列A).cuda(device)
                图片_分_torch = torch.from_numpy(图片_分_枝).cuda(device)
                目标输出_分_torch = torch.from_numpy(目标输出_分_枝).cuda(device)
                src_mask, trg_mask = create_masks(操作_分_torch, 操作_分_torch, device)
                if 图片_分_torch.shape[0] != 操作_分_torch.shape[0]:
                    continue
                状态 = {'操作序列': 操作_分_枝, '图片张量': 图片_分_枝, 'trg_mask': trg_mask}
                动作, 动作可能性, 评价 = 智能体.选择动作批量(状态, device, 目标输出_分_torch, True)
                实际输出, _ = model_判断状态(图片_分_torch, 操作序列A_torch, trg_mask)
                _, 抽样 = torch.topk(实际输出, k=1, dim=-1)
                抽样np = 抽样.cpu().numpy()
                回报 = np.ones_like(抽样np[0, :, 0])
                回报 = 回报.astype(np.float32)
                for 计数 in range(抽样np.shape[1]):
                    状况 = 状态列表[抽样np[0, 计数, 0]]
                    得分 = 状态词典[状况]
                    回报[计数] = 得分
                智能体.监督强化学习(device, 状态, 回报, 动作, 动作可能性, 评价)
                if 计数 % 1 == 0:
                    total_seconds(time_start)
                计数 += 1
                i += 1
    total_seconds(time_start)
    智能体.保存模型(j + 1)
