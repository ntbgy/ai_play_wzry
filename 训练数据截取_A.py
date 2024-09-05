import os
import threading
import time

import torchvision
from pynput import keyboard
from pynput.keyboard import Key, Listener

from Batch import create_masks
from resnet_utils import myResnet
from 取训练数据 import *
from 杂项 import *
from 模型_策略梯度 import 智能体
from 辅助功能 import 状态信息综合
from 运行辅助 import *

# 雷电模拟器
_DEVICE_ID = 'emulator-5554'
窗口名称 = "LIO-AN00"

训练数据保存目录 = '../训练数据样本/未用'
if not os.path.exists(训练数据保存目录):
    os.makedirs(训练数据保存目录)
lock = threading.Lock()
start = time.time()
end = time.time()
fun_start = 0
time_interval = 0
index = 0
数据字典 = {'interval_times': 0, 'max_interval': 0., 'interval_location': []}
count = 0
count_dict = {'first_time': 0., 'first_p_to_second_r': 0.}
keyBoard_dict = {'Key.enter': '\n',
                 'Key.space': ' ',
                 "Key.tab": '\t'}

W键按下 = False
S键按下 = False
A键按下 = False
D键按下 = False
Q键按下 = False
攻击态 = False
手动模式 = False
攻击放开 = True
AI打开 = True
操作列 = []
自动 = 0

N = 15000  # 运行N次后学习
条数 = 100
轮数 = 3
学习率 = 0.0003
智能体 = 智能体(动作数=7, 并行条目数=条数,
                学习率=学习率, 轮数=轮数,
                输入维度=6)


def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):

        return key.char
    else:

        return str(key)


# 监听按压
def on_press(key):
    global fun_start, time_interval, index, 数据字典, count, count_dict, W键按下, S键按下, A键按下, D键按下, 手动模式, 操作列, AI打开, 攻击放开, Q键按下, 攻击态

    key_name = get_key_name(key)
    操作 = ''
    if key_name == 'w':
        W键按下 = True
    elif key_name == 'a':
        A键按下 = True
    elif key_name == 's':
        S键按下 = True
    elif key_name == 'd':
        D键按下 = True
    elif key_name == 'q':
        Q键按下 = True
    elif key_name == 'i':
        AI打开 = bool(1 - AI打开)

    elif key_name == 'Key.space':
        操作 = '召唤师技能'
    elif key_name == 'Key.end':
        操作 = '补刀'
    elif key_name == 'Key.page_down':
        操作 = '推塔'
    elif key_name == 'j':
        操作 = '一技能'
    elif key_name == 'k':
        操作 = '二技能'
    elif key_name == 'l':
        操作 = '三技能'
    elif key_name == 'f':
        操作 = '回城'
    elif key_name == 'g':
        操作 = '恢复'
    elif key_name == 'h':
        操作 = '召唤师技能'
    elif key_name == 'Key.left':
        操作 = '一技能'
    elif key_name == 'Key.down':
        操作 = '二技能'
    elif key_name == 'Key.right':
        操作 = '三技能'
    elif key_name == 'Key.up':
        攻击态 = True

    lock.acquire()
    if 操作 != '':
        操作列.append(操作)
    lock.release()
    # print("正在按压:", key_name)


# 监听释放
def on_release(key):
    global start, fun_start, time_interval, index, count, count_dict, W键按下, S键按下, A键按下, D键按下, 攻击放开, Q键按下, 攻击态

    key_name = get_key_name(key)
    if key_name == 'w':
        W键按下 = False
    elif key_name == 'a':
        A键按下 = False
    elif key_name == 's':
        S键按下 = False
    elif key_name == 'd':
        D键按下 = False
    elif key_name == 'q':
        Q键按下 = False

    elif key_name == 'Key.up':

        攻击态 = False
    print("已经释放:", key_name)
    if key == Key.esc:
        # 停止监听
        return False


# 开始监听
def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def 处理方向():
    # W键按下 = False
    # S键按下 = False
    # A键按下 = False
    # D键按下 = False
    if Q键按下 == True:
        return ('移动停')
    elif W键按下 == True and S键按下 == False and A键按下 == False and D键按下 == False:
        return ('上移')
    elif W键按下 == False and S键按下 == True and A键按下 == False and D键按下 == False:
        return ('下移')
    elif W键按下 == False and S键按下 == False and A键按下 == True and D键按下 == False:
        return ('左移')
    elif W键按下 == False and S键按下 == False and A键按下 == False and D键按下 == True:
        return ('右移')
    elif W键按下 == True and S键按下 == False and A键按下 == True and D键按下 == False:
        return ('左上移')
    elif W键按下 == True and S键按下 == False and A键按下 == False and D键按下 == True:
        return ('右上移')
    elif W键按下 == False and S键按下 == True and A键按下 == True and D键按下 == False:
        return ('左下移')
    elif W键按下 == False and S键按下 == True and A键按下 == False and D键按下 == True:
        return ('右下移')
    else:
        return ('')


加三技能 = 'd 0 552 1878 100\nc\nu 0\nc\n'
加二技能 = 'd 0 446 1687 100\nc\nu 0\nc\n'
加一技能 = 'd 0 241 1559 100\nc\nu 0\nc\n'
购买 = 'd 0 651 207 100\nc\nu 0\nc\n'
词数词典路径 = "./json/词_数表.json"
数_词表路径 = "./json/数_词表.json"
操作查询路径 = "./json/名称_操作.json"
操作词典 = {"图片号": "0", "移动操作": "无移动", "动作操作": "无动作"}
th = threading.Thread(target=start_listen, )
th.start()  # 启动线程

if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
    词_数表, 数_词表 = 读出引索(词数词典路径, 数_词表路径)
else:
    raise ValueError("词_数表, 数_词表 获取失败！")
with open(词数词典路径, encoding='utf8') as f:
    词数词典 = json.load(f)
with open(操作查询路径, encoding='utf8') as f:
    操作查询词典 = json.load(f)

方向表 = ['上移', '下移', '左移', '右移', '左上移', '左下移', '右上移', '右下移']

设备 = MyMNTDevice(_DEVICE_ID)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
resnet101 = myResnet(mod)

while True:
    txt = os
    if not AI打开:
        continue
    图片路径 = 训练数据保存目录 + '/{}/'.format(str(int(time.time())))
    os.mkdir(图片路径)

    记录文件 = open(图片路径 + '_操作数据.json', 'w+')

    图片张量 = torch.Tensor(0)
    操作张量 = torch.Tensor(0)

    伪词序列 = torch.from_numpy(np.ones((1, 60)).astype(np.int64)).cuda(device).unsqueeze(0)

    指令延时 = 0

    操作序列 = np.ones((1,))
    操作序列[0] = 128
    计数 = 0
    time_start = time.time()
    旧指令 = '移动停'
    for i in range(100000):
        if not AI打开:
            break
        try:
            imgA = 取图(窗口名称)
        except:
            AI打开 = False
            print('取图失败')
            break

        计时开始 = time.time()

        if 图片张量.shape[0] == 0:

            img = np.array(imgA)

            img = torch.from_numpy(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
            _, out = resnet101(img)
            图片张量 = out.reshape(1, 6 * 6 * 2048)

        elif 图片张量.shape[0] < 300:

            img = np.array(imgA)

            img = torch.from_numpy(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
            _, out = resnet101(img)
            图片张量 = torch.cat((图片张量, out.reshape(1, 6 * 6 * 2048)), 0)
            操作序列 = np.append(操作序列, 动作)

        else:

            img = np.array(imgA)

            img = torch.from_numpy(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
            _, out = resnet101(img)
            图片张量 = 图片张量[1:300, :]
            操作序列 = 操作序列[1:300]
            操作序列 = np.append(操作序列, 动作)
            图片张量 = torch.cat((图片张量, out.reshape(1, 6 * 6 * 2048)), 0)

        操作张量 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(操作张量.unsqueeze(0), 操作张量.unsqueeze(0), device)

        状态 = 状态信息综合(图片张量.cpu().numpy(), 操作序列, trg_mask)

        动作, 动作可能性, 评价 = 智能体.选择动作(状态, device, 1, False)
        LI = 操作张量.contiguous().view(-1)
        if 计数 % 50 == 0 and 计数 != 0:
            设备.发送(购买)
            设备.发送(加三技能)
            设备.发送(加二技能)
            设备.发送(加一技能)
            设备.发送(操作查询词典['移动停'])
            print(旧指令, '周期')
            time.sleep(0.02)
            设备.发送(操作查询词典[旧指令])

        if 计数 % 1 == 0:
            time_end = time.time()

            指令 = 数_词表[str(动作)]
            指令集 = 指令.split('_')

            # 操作词典 = {"图片号": "0", "移动操作": "无移动", "动作操作": "无动作"}
            操作词典['图片号'] = str(i)
            方向结果 = 处理方向()
            if 方向结果 != '' or len(操作列) != 0 or 攻击态 == True:
                if 方向结果 == '':
                    操作词典['移动操作'] = 指令集[0]
                else:
                    操作词典['移动操作'] = 方向结果

                if len(操作列) != 0:
                    操作词典['动作操作'] = 操作列[0]
                    lock.acquire()
                    del 操作列[0]
                    lock.release()
                elif 攻击态 == True:
                    操作词典['动作操作'] = '攻击'

                else:
                    操作词典['动作操作'] = '无动作'

                路径_a = 图片路径 + '{}.jpg'.format(str(i))
                imgA.save(路径_a)
                if 自动 == 0:
                    操作词典['结束'] = 1
                else:
                    操作词典['结束'] = 0
                自动 = 1
                json.dump(操作词典, 记录文件, ensure_ascii=False)
                记录文件.write('\n')

                新指令 = 操作词典['移动操作']
                if 新指令 != 旧指令 and 新指令 != '无移动':
                    旧指令 = 新指令
                    # print(旧指令,操作查询词典[旧指令])
                    try:
                        print('手动模式', 旧指令)

                        设备.发送(操作查询词典[旧指令])

                    except:
                        AI打开 = False
                        print('发送失败')
                        break

                    time.sleep(0.01)

                if 操作词典['动作操作'] != '无动作' and 操作词典['动作操作'] != '发起集合' and 操作词典[
                    '动作操作'] != '发起进攻' and 操作词典['动作操作'] != '发起撤退':
                    print('手动', 指令集[1])
                    try:
                        设备.发送(操作查询词典[操作词典['动作操作']])
                    except:
                        AI打开 = False
                        print('发送失败')
                        break
            else:
                操作列 = []
                操作词典['移动操作'] = 指令集[0]
                操作词典['动作操作'] = 指令集[1]

                新指令 = 指令集[0]
                if 新指令 != 旧指令 and 新指令 != '无移动':
                    旧指令 = 新指令
                    # print(旧指令,操作查询词典[旧指令])
                    try:
                        print(旧指令)

                        设备.发送(操作查询词典[旧指令])

                    except:
                        AI打开 = False
                        print('发送失败')
                        break

                    time.sleep(0.01)
                路径_a = 图片路径 + '{}.jpg'.format(str(i))
                imgA.save(路径_a)
                自动 = 0
                操作词典['结束'] = 0
                json.dump(操作词典, 记录文件, ensure_ascii=False)
                记录文件.write('\n')

                新指令 = 操作词典['移动操作']
                if 指令集[1] != '无动作' and 指令集[1] != '发起集合' and 指令集[1] != '发起进攻' and 指令集[
                    1] != '发起撤退':
                    print(指令集[1])
                    try:
                        设备.发送(操作查询词典[指令集[1]])
                    except:
                        AI打开 = False
                        print('发送失败')
                        break
            用时1 = 0.22 - (time.time() - 计时开始)
            if 用时1 > 0:
                time.sleep(用时1)

            # print(用时1)
            用时 = time_end - time_start
            # print("用时{} 第{}张 延时{}".format(用时, i,用时1),'A键按下', A键按下, 'W键按下', W键按下, 'S键按下', S键按下, 'D键按下', D键按下, '旧指令', 旧指令, 'AI打开', AI打开, '操作列', 操作列)

            计数 = 计数 + 1
            if i % 3000 == 0:
                # AI打开 = False
                # import pygame

                # pygame.mixer.init()
                # pygame.mixer.music.load('G:/AS.mp3')
                # pygame.mixer.music.set_volume(0.2)
                # pygame.mixer.music.play()
                print("此处可有音乐")
                time.sleep(1)

    记录文件.close()
    time.sleep(1)
    print('AI打开', AI打开)
