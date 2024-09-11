import sys
import threading
import time

import pyautogui
import torchvision
from airtest.core.api import *
from pynput import keyboard
from pynput.keyboard import Key, Listener
from pywinauto import Application

from Batch import create_masks
from common.airtestProjectsCommon import get_img_txt
from common.env import training_data_save_directory, project_root_path, 状态词典, 状态词典B, 操作查询词典, 操作词典
from common.my_logger import logger
from common.stop import stop
from resnet_utils import myResnet
from 取训练数据 import *
from 杂项 import *
from 模型_策略梯度 import 智能体
from 辅助功能 import 状态信息综合
from 运行辅助 import *

# threading.Lock是 Python 中threading模块提供的一种简单的线程同步机制，用于实现互斥锁（Mutex Lock）。
# 当一个线程获取了锁（通过lock.acquire()方法）后，其他线程在尝试获取该锁时将被阻塞，直到锁被释放（通过lock.release()方法）。
lock = threading.Lock()
# 全局变量初始化
AI打开 = True
攻击态 = False
W键按下 = False
S键按下 = False
A键按下 = False
D键按下 = False
Q键按下 = False
操作列 = []
sp = stop()
智能体 = 智能体(
    动作数=7,
    并行条目数=100,
    学习率=0.0003,
    轮数=3,
    输入维度=6
)

def get_key_name(key) -> str:
    """从pynput.keyboard模块中的按键对象中提取出一个可识别的键名"""
    if isinstance(key, keyboard.KeyCode):
        return key.char
    else:
        return str(key)


# 监听按压
def on_press(key):
    global W键按下, S键按下, A键按下, D键按下, 操作列, Q键按下, 攻击态
    # 哎，.lower() 这里给自己坑死了
    key_name = get_key_name(key).lower()
    manual_manipulation = ''
    lock.acquire()
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
    elif key_name == 'j':
        manual_manipulation = '一技能'
    elif key_name == 'k':
        manual_manipulation = '二技能'
    elif key_name == 'l':
        manual_manipulation = '三技能'
    elif key_name == 'f':
        manual_manipulation = '回城'
    elif key_name == 'g':
        manual_manipulation = '恢复'
    elif key_name == 'h':
        manual_manipulation = '召唤师技能'
    elif key_name == 'key.left':
        manual_manipulation = '一技能'
    elif key_name == 'key.down':
        manual_manipulation = '二技能'
    elif key_name == 'key.right':
        manual_manipulation = '三技能'
    elif key_name == 'key.space':
        manual_manipulation = '召唤师技能'
    elif key_name == 'key.up':
        攻击态 = True
    elif key_name == 'key.delete':
        manual_manipulation = '补刀'
    elif key_name == 'key.end':
        manual_manipulation = '攻击'
    elif key_name == 'key.page_down':
        manual_manipulation = '推塔'
    if manual_manipulation in ('回城', '恢复'):
        操作列.clear()
        操作列.append(manual_manipulation)
    elif manual_manipulation != '':
        操作列.append(manual_manipulation)
    lock.release()


# 监听释放
def on_release(key):
    global W键按下, S键按下, A键按下, D键按下, Q键按下, 攻击态
    key_name = get_key_name(key)
    if key == Key.esc:
        logger.warning('停止监听')
        return False
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
    elif key_name=='Key.up' :
        攻击态=False


# 开始监听
def start_listen():
    # noinspection PyTypeChecker
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def 处理方向():
    if Q键按下 is True:
        return '移动停'
    elif W键按下 is True and S键按下 is False and A键按下 is False and D键按下 is False:
        return '上移'
    elif W键按下 is False and S键按下 is True and A键按下 is False and D键按下 is False:
        return '下移'
    elif W键按下 is False and S键按下 is False and A键按下 is True and D键按下 is False:
        return '左移'
    elif W键按下 is False and S键按下 is False and A键按下 is False and D键按下 is True:
        return '右移'
    elif W键按下 is True and S键按下 is False and A键按下 is True and D键按下 is False:
        return '左上移'
    elif W键按下 is True and S键按下 is False and A键按下 is False and D键按下 is True:
        return '右上移'
    elif W键按下 is False and S键按下 is True and A键按下 is True and D键按下 is False:
        return '左下移'
    elif W键按下 is False and S键按下 is True and A键按下 is False and D键按下 is True:
        return '右下移'
    else:
        return ''


def 判断状态(device, resnet101, image_path):
    from 模型_策略梯度 import Transformer
    model_判断状态 = Transformer(6, 768, 2, 12, 0.0, 6 * 6 * 2048)
    model_判断状态.load_state_dict(torch.load('E:/ai-play-wzry/weights/model_weights_judgment_state.pth'))
    model_判断状态.cuda(device)
    image = Image.open(image_path)
    图片数组 = np.asarray(image)
    截屏 = torch.from_numpy(图片数组).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
    _, out = resnet101(截屏)
    out = torch.reshape(out, (1, 6 * 6 * 2048))
    操作序列A = np.ones((1, 1))
    操作张量A = torch.from_numpy(操作序列A.astype(np.int64)).cuda(device)
    src_mask, trg_mask = create_masks(操作张量A.unsqueeze(0), 操作张量A.unsqueeze(0), device)
    outA = out.detach()
    实际输出, _ = model_判断状态(outA.unsqueeze(0), 操作张量A.unsqueeze(0), trg_mask)
    _, 抽样 = torch.topk(实际输出, k=1, dim=-1)
    抽样np = 抽样.cpu().numpy()
    状态列表 = []
    for K in 状态词典B:
        状态列表.append(K)
    状况 = 状态列表[抽样np[0, 0, 0, 0]]
    得分 = 状态词典[状况]
    logger.debug(f'image_path: {image_path}，判断状态：{状况}，得分：{得分}')
    return 状况, 得分


def 追加记录(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')


def 发送指令(pyminitouch_device, 指令, sleep_time=0):
    global AI打开
    try:
        pyminitouch_device.发送(操作查询词典[指令])
        time.sleep(sleep_time)
        return True
    except:
        AI打开 = False
        logger.error('发送失败')
        return False


def 训练数据截取(device_id, scrcpy_windows_name, 手工介入=True, flag_file_name='stop_flag.txt'):
    global 操作列, sp, AI打开, 攻击态
    AI打开 = True
    自动 = 0
    if 手工介入 is True:
        th = threading.Thread(target=start_listen, )
        th.daemon = True
        th.start()  # 启动线程
    词数词典路径 = f"{project_root_path}/json/词_数表.json"
    数_词表路径 = f"{project_root_path}/json/数_词表.json"

    if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
        词_数表, 数_词表 = 读出引索(词数词典路径, 数_词表路径)
    else:
        raise ValueError("词_数表, 数_词表 获取失败！")

    # with open(词数词典路径, encoding='utf8') as f:
    #     词数词典 = json.load(f)
    pyminitouch_device = MyMNTDevice(device_id)

    # 加载预训练的 ResNet - 101 模型
    # 模型设置为评估模式
    # 将模型移动到指定的 GPU 设备上
    # 关闭梯度计算
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
    resnet101 = myResnet(mod)

    try:
        while True:
            save_image_directory = training_data_save_directory + '/{}/'.format(str(int(time.time())))
            os.mkdir(save_image_directory)
            save_file_path = save_image_directory + '_操作数据.json'
            with open(save_file_path, 'w', encoding='utf-8') as f:
                f.write('')
            图片张量 = torch.Tensor(0)
            操作序列 = np.ones((1,))
            操作序列[0] = 128
            计数 = 0
            旧移动指令 = '移动停'
            for i in range(6 * 1000):
                if AI打开 is False:
                    break
                try:
                    imgA = get_window_image(scrcpy_windows_name)
                except:
                    AI打开 = False
                    logger.error('取图失败！')
                    break

                计时开始 = time.time()
                # 动作 = None
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

                指令 = 数_词表[str(动作)]
                logger.debug(f"指令：{指令}，动作可能性：{动作可能性}，评价：{评价}。")
                指令集 = 指令.split('_')
                自动移动指令, 自动动作指令 = 指令集[0], 指令集[1]

                # 游戏中可以设置自动购买装备，自动加技能
                # if 计数 % 50 == 0 and 计数 != 0:
                #     pyminitouch_device.发送(操作查询词典['购买'])
                #     pyminitouch_device.发送(操作查询词典['加一技能'])
                #     pyminitouch_device.发送(操作查询词典['加二技能'])
                #     pyminitouch_device.发送(操作查询词典['加三技能'])
                #     # pyminitouch_device.发送(操作查询词典['移动停'])
                #     time.sleep(0.01)

                save_image_path = save_image_directory + '{}.jpg'.format(str(i))
                imgA.save(save_image_path)
                sp.set_image_path(save_image_path)

                操作词典['图片号'] = str(i)
                手动移动指令 = 处理方向()

                if 手动移动指令 != '' or len(操作列) != 0 or 攻击态 == True:
                    if 手动移动指令 == '':
                        操作词典['移动操作'] = 自动移动指令
                    else:
                        操作词典['移动操作'] = 手动移动指令

                    if len(操作列) != 0:
                        操作词典['动作操作'] = 操作列[0]
                        lock.acquire()
                        del 操作列[0]
                        lock.release()
                    elif 攻击态 is True:
                        操作词典['动作操作'] = '攻击'
                    else:
                        操作词典['动作操作'] = '无动作'

                    if 自动 == 0:
                        操作词典['结束'] = 1
                    else:
                        操作词典['结束'] = 0

                    自动 = 1
                    追加记录(save_file_path, 操作词典)

                    新移动指令 = 操作词典['移动操作']
                    if 新移动指令 != 旧移动指令 and 新移动指令 != '无移动':
                        logger.info(f"移动操作，手动模式，{旧移动指令} -> {新移动指令}")
                        if not 发送指令(pyminitouch_device, 新移动指令, 0.01):
                            break
                        # 连续移动，保持就好了，否则会鬼畜
                        旧移动指令 = 新移动指令

                    新动作指令 = 操作词典['动作操作']
                    if 新动作指令 not in ('无动作', '发起集合', '发起进攻', '发起撤退'):
                        logger.info(f"动作操作，手动模式，{自动动作指令} -> {新动作指令}")
                        if not 发送指令(pyminitouch_device, 新动作指令):
                            break

                elif 手动移动指令 == '' and len(操作列) == 0:
                    操作列 = []
                    操作词典['移动操作'] = 自动移动指令
                    操作词典['动作操作'] = 自动动作指令
                    if 自动移动指令 != 旧移动指令 and 自动移动指令 != '无移动':
                        if not 发送指令(pyminitouch_device, 自动移动指令, 0.01):
                            break
                    # 连续移动，保持就好了，否则会鬼畜
                    旧移动指令 = 自动移动指令

                    自动 = 0
                    操作词典['结束'] = 0
                    追加记录(save_file_path, 操作词典)
                    if 自动动作指令 not in ('无动作', '发起集合', '发起进攻', '发起撤退'):
                        if not 发送指令(pyminitouch_device, 自动动作指令):
                            break

                计数 += 1

                用时 = 0.22 - (time.time() - 计时开始)
                if 用时 > 0:
                    time.sleep(用时)

                if (sp.get_stop() is True
                        or os.path.exists(flag_file_name)):
                    pyminitouch_device.stop()
                    break

            time.sleep(1)
            # 检查标记文件
            if os.path.exists(flag_file_name):
                os.remove(flag_file_name)
                logger.debug("删除标记文件")
                break
            elif sp.get_stop() is True:
                break
            elif AI打开 is False:
                logger.info(('AI打开', AI打开))
                continue
        pyautogui.press('esc')
    except KeyboardInterrupt:
        logger.warning("用户中断了程序的运行")
        pyminitouch_device.stop()


# 检测游戏是否结束
def check_game_status(flag_file_name):
    global sp
    time.sleep(6 * 60)
    logger_ppocr = logging.getLogger("ppocr")
    logger_ppocr.setLevel(logging.ERROR)
    while True:
        if os.path.exists(flag_file_name):
            return
        image_path = sp.get_image_path()
        if image_path is None:
            continue
        txt = get_img_txt(image_path)
        keywords = ['返回大厅', '再来一局', '继续', '胜利', '失败', '请选择']
        if any(keyword in txt for keyword in keywords):
            logger.info("检测到游戏结束")
            sp.set_stop(True)
            break
        time.sleep(60)

def scrcpy(s='--max-size 960 --always-on-top'):
    """
    启动scrcpy
    """
    os.system(f'scrcpy {s}')

# 启动游戏
# noinspection PyUnresolvedReferences
def start_game(dir_path):
    """
    "android:///"
    "android://127.0.0.1:5037/emulator-5554"
    "android://127.0.0.1:5037/emulator-5556"
    """
    sys.path.append('auto/王者荣耀/对战.air')
    using('auto/王者荣耀/对战.air')
    from 对战 import 离线5V5
    logger.debug('start_game')
    离线5V5(dir_path)
    os.chdir(dir_path)


def single_run(dir_path, device_id, scrcpy_windows_name, flag_file_name):
    """
    运行单局游戏
    """
    # 防止还没开始就结束了
    global sp
    sp.set_stop(False)
    if os.path.exists(flag_file_name):
        os.remove(flag_file_name)
    # 进入游戏
    start_game(dir_path)
    # 打开scrcpy
    th1 = threading.Thread(target=scrcpy)
    # 将线程设置为守护线程（daemon = True），当主线程结束时，守护线程会自动退出。
    th1.daemon = True
    th1.start()
    # AI打游戏
    th2 = threading.Thread(target=训练数据截取, args=(device_id, scrcpy_windows_name, True, flag_file_name))
    th2.start()
    # 检测游戏是否结束，不停检测会卡死，emmmm
    th3 = threading.Thread(target=check_game_status, args=(flag_file_name,))
    th3.start()
    th2.join()
    logger.info('AI打游戏线程done')
    th3.join()
    logger.info('检测游是否已结束线程done')
    app = Application(backend="uia").connect(title=scrcpy_windows_name)
    main_window = app.window(title_re=scrcpy_windows_name)
    main_window.close()
    logger.info('关闭scrcpy窗口done')


if __name__ == '__main__':
    import logging
    import os
    import threading

    from airtest.cli.parser import cli_setup
    from airtest.core.api import auto_setup

    from common.airtestProjectsCommon import clean_log

    # 设置日志级别
    logger_airtest = logging.getLogger("airtest")
    logger_ppocr = logging.getLogger("ppocr")
    logger_ppocr.setLevel(logging.ERROR)
    logger_airtest.setLevel(logging.ERROR)
    # 清空存量日志
    clean_log()

    dir_path = os.path.dirname(os.path.abspath(__file__))
    device_id = 'emulator-5554'
    scrcpy_windows_name = "LIO-AN00"
    flag_file_name = 'stop_flag.txt'
    airtest_devices = "android:///"
    # 连接设备
    if not cli_setup():
        auto_setup(
            __file__,
            logdir=True,
            devices=[airtest_devices]
        )
    """
    # AI打游戏
    th2 = threading.Thread(target=训练数据截取, args=(device_id, scrcpy_windows_name, True, flag_file_name))
    th2.start()
    # 检测游戏是否结束，不停检测会卡死，emmmm
    th3 = threading.Thread(target=check_game_status, args=(flag_file_name,))
    th3.start()
    th2.join()
    logger.info('AI打游戏线程done')
    th3.join()
    logger.info('检测游是否已结束线程done')
    """
    for i in range(1, 6):
        logger.warning(f'第{i}局游戏开始！')
        try:
            single_run(dir_path, device_id, scrcpy_windows_name, flag_file_name)
            time.sleep(5)
        except:
            pass
        logger.warning(f'第{i}局游戏结束！')
