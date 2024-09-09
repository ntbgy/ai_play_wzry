import sys
import threading
from pathlib import Path

import torchvision
from airtest.core.api import *
from pynput import keyboard
from pynput.keyboard import Key, Listener

from Batch import create_masks
from common.airtestProjectsCommon import get_img_txt
from common.env import training_data_save_directory, project_root_path
from common.my_logger import logger
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
W键按下 = False
S键按下 = False
A键按下 = False
D键按下 = False
Q键按下 = False
操作列 = []
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
    global W键按下, S键按下, A键按下, D键按下, \
        操作列, Q键按下
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
    # elif key_name == 'i':
    #     AI打开 = bool(1 - AI打开)
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
        # 待定
        pass
    elif key_name == 'key.up':
        # 暂定攻击
        manual_manipulation = '攻击'
    elif key_name == 'key.delete':
        manual_manipulation = '补刀'
    elif key_name == 'key.end':
        manual_manipulation = '攻击'
    elif key_name == 'key.page_down':
        manual_manipulation = '推塔'

    if manual_manipulation in ('回城', '恢复'):
        操作列.insert(0, manual_manipulation)
    elif manual_manipulation != '':
        操作列.append(manual_manipulation)
    lock.release()


# 监听释放
def on_release(key):
    global W键按下, S键按下, A键按下, D键按下, \
        Q键按下
    key_name = get_key_name(key)
    if key == Key.esc:
        logger.info('停止监听')
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


# 开始监听
def start_listen():
    # noinspection PyTypeChecker
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def 处理方向():
    if Q键按下 is True:
        return '移动停'
    elif W键按下 == True and S键按下 == False and A键按下 == False and D键按下 == False:
        return '上移'
    elif W键按下 == False and S键按下 == True and A键按下 == False and D键按下 == False:
        return '下移'
    elif W键按下 == False and S键按下 == False and A键按下 == True and D键按下 == False:
        return '左移'
    elif W键按下 == False and S键按下 == False and A键按下 == False and D键按下 == True:
        return '右移'
    elif W键按下 == True and S键按下 == False and A键按下 == True and D键按下 == False:
        return '左上移'
    elif W键按下 == True and S键按下 == False and A键按下 == False and D键按下 == True:
        return '右上移'
    elif W键按下 == False and S键按下 == True and A键按下 == True and D键按下 == False:
        return '左下移'
    elif W键按下 == False and S键按下 == True and A键按下 == False and D键按下 == True:
        return '右下移'
    else:
        return ''


def 训练数据截取(device_id, scrcpy_windows_name, 手工介入=True, flag_file_name='stop_flag.txt'):
    手动模式 = False
    AI打开 = True
    自动 = 0
    global 操作列
    if 手工介入 is True:
        th = threading.Thread(target=start_listen, )
        th.start()  # 启动线程
    词数词典路径 = f"{project_root_path}/json/词_数表.json"
    数_词表路径 = f"{project_root_path}/json/数_词表.json"
    操作查询路径 = f"{project_root_path}/json/名称_操作.json"
    操作词典 = {"图片号": "0", "移动操作": "无移动", "动作操作": "无动作"}
    if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
        词_数表, 数_词表 = 读出引索(词数词典路径, 数_词表路径)
    else:
        raise ValueError("词_数表, 数_词表 获取失败！")
    with open(词数词典路径, encoding='utf8') as f:
        词数词典 = json.load(f)
    with open(操作查询路径, encoding='utf8') as f:
        操作查询词典 = json.load(f)
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
            图片路径 = training_data_save_directory + '/{}/'.format(str(int(time.time())))
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
            for i in range(100 * 1000):
                # 检查标记文件
                if os.path.exists(flag_file_name):
                    break
                if AI打开 is False:
                    break
                try:
                    imgA = get_window_image(scrcpy_windows_name)
                except:
                    AI打开 = False
                    logger.info('取图失败！')
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
                    # noinspection PyUnboundLocalVariable
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
                    pyminitouch_device.发送(操作查询词典['购买'])
                    pyminitouch_device.发送(操作查询词典['加一技能'])
                    pyminitouch_device.发送(操作查询词典['加二技能'])
                    pyminitouch_device.发送(操作查询词典['加三技能'])
                    pyminitouch_device.发送(操作查询词典['移动停'])
                    time.sleep(0.02)
                    logger.info((旧指令, '周期'))
                    pyminitouch_device.发送(操作查询词典[旧指令])
                路径_a = 图片路径 + '{}.jpg'.format(str(i))
                if 计数 % 1 == 0:
                    time_end = time.time()
                    指令 = 数_词表[str(动作)]
                    指令集 = 指令.split('_')
                    操作词典['图片号'] = str(i)
                    方向结果 = 处理方向()
                    if 方向结果 != '' or len(操作列) != 0:
                        if 方向结果 == '':
                            操作词典['移动操作'] = 指令集[0]
                        else:
                            操作词典['移动操作'] = 方向结果
                        if len(操作列) != 0:
                            操作词典['动作操作'] = 操作列[0]
                            lock.acquire()
                            del 操作列[0]
                            lock.release()
                        else:
                            操作词典['动作操作'] = '无动作'
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
                            try:
                                logger.info(('手动模式', 旧指令))
                                pyminitouch_device.发送(操作查询词典[旧指令])
                            except:
                                AI打开 = False
                                logger.info('发送失败')
                                break
                            time.sleep(0.01)
                        if (操作词典['动作操作'] != '无动作'
                                and 操作词典['动作操作'] != '发起集合'
                                and 操作词典['动作操作'] != '发起进攻'
                                and 操作词典['动作操作'] != '发起撤退'):
                            logger.debug(指令集)
                            logger.info(('手动', 指令集[1]))
                            try:
                                pyminitouch_device.发送(操作查询词典[操作词典['动作操作']])
                            except:
                                AI打开 = False
                                logger.info('发送失败')
                                break
                    else:
                        操作列 = []
                        操作词典['移动操作'] = 指令集[0]
                        操作词典['动作操作'] = 指令集[1]
                        新指令 = 指令集[0]
                        if 新指令 != 旧指令 and 新指令 != '无移动':
                            旧指令 = 新指令
                            try:
                                logger.info(旧指令)
                                pyminitouch_device.发送(操作查询词典[旧指令])
                            except:
                                AI打开 = False
                                logger.info('发送失败')
                                break
                            time.sleep(0.01)
                        imgA.save(路径_a)
                        自动 = 0
                        操作词典['结束'] = 0
                        json.dump(操作词典, 记录文件, ensure_ascii=False)
                        记录文件.write('\n')
                        新指令 = 操作词典['移动操作']
                        if 指令集[1] != '无动作' and 指令集[1] != '发起集合' and 指令集[1] != '发起进攻' and 指令集[
                            1] != '发起撤退':
                            logger.info(指令集[1])
                            try:
                                pyminitouch_device.发送(操作查询词典[指令集[1]])
                            except:
                                AI打开 = False
                                logger.info('发送失败')
                                break
                    用时1 = 0.22 - (time.time() - 计时开始)
                    if 用时1 > 0:
                        time.sleep(用时1)
                    用时 = time_end - time_start
                    计数 = 计数 + 1
            记录文件.close()
            time.sleep(1)
            # 检查标记文件
            if os.path.exists(flag_file_name):
                pyminitouch_device.stop()
                os.remove(flag_file_name)
                logger.debug("删除标记文件")
                break
            if AI打开 is False:
                continue
            logger.info(('AI打开', AI打开))
    except KeyboardInterrupt:
        logger.warning("用户中断了程序的运行")
        pyminitouch_device.stop()
        try:
            # noinspection PyUnboundLocalVariable
            记录文件.close()
        except NameError as e:
            logger.warning(e)
        time.sleep(0.5)


# 检测游戏是否结束
def check_game_status(dir_path, scrcpy_windows_name, flag_file_name='stop_flag.txt'):
    logger_ppocr = logging.getLogger("ppocr")
    logger_ppocr.setLevel(logging.ERROR)
    game_running = True
    while game_running:
        image = get_window_image(scrcpy_windows_name)
        image_name = 'window_screenshot.png'
        image.save(image_name)
        txt = get_img_txt(str(
            Path(dir_path) / Path(image_name)))
        if ('返回大厅' in txt
                or '再来一局' in txt
                or '继续' in txt
                or '胜利' in txt
                or '失败' in txt
        ):
            game_running = False
            logger.info("检测到游戏结束")
            logger.debug('创建标记文件')
            with open(flag_file_name, 'w') as f:
                f.write('stop')
            os.remove(image_name)


def 训练():
    os.system(
        r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe C:\Users\ntbgy\PycharmProjects\ai-play-wzry\02处理训练数据.py')
    time.sleep(5)
    os.system(
        r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe C:\Users\ntbgy\PycharmProjects\ai-play-wzry\03训练主模型.py')
    logger.info('done')


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
    # 防止还没开始就结束了
    if os.path.exists(flag_file_name):
        os.remove(flag_file_name)

    # 进入游戏
    start_game(dir_path)

    # 启动AI打游戏
    th1 = threading.Thread(target=训练数据截取, args=(
        device_id, scrcpy_windows_name, False, flag_file_name))
    th1.start()

    # 检测游戏是否结束
    # time.sleep(3 * 60)
    th2 = threading.Thread(target=check_game_status, args=(
        dir_path, scrcpy_windows_name, flag_file_name))
    th2.start()

    th1.join()
    th2.join()

    return True


if __name__ == '__main__':
    # import logging
    # import os
    # import threading
    #
    # from airtest.cli.parser import cli_setup
    # from airtest.core.api import auto_setup
    #
    # from common.airtestProjectsCommon import clean_log
    #
    # # 设置日志级别
    # logger_airtest = logging.getLogger("airtest")
    # logger_ppocr = logging.getLogger("ppocr")
    # logger_ppocr.setLevel(logging.ERROR)
    # logger_airtest.setLevel(logging.ERROR)
    # # 清空存量日志
    # clean_log()
    #
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    # device_id = 'emulator-5554'
    # scrcpy_windows_name = "LIO-AN00"
    # flag_file_name = 'stop_flag_1.txt'
    # airtest_devices = "android:///"
    # # 连接设备
    # if not cli_setup():
    #     auto_setup(
    #         __file__,
    #         logdir=True,
    #         devices=[airtest_devices]
    #     )
    #
    #
    # def scrcpy():
    #     os.system('scrcpy --max-size 960')
    #
    #
    # th1 = threading.Thread(target=scrcpy)
    # th1.start()
    #
    # res = single_run(dir_path, device_id, scrcpy_windows_name, flag_file_name)
    # if res is True:
    #     from pywinauto.application import Application
    #
    #     app = Application(backend="uia").connect(title=scrcpy_windows_name)
    #     main_window = app.window(title_re=scrcpy_windows_name)
    #     main_window.close()

    device_id = 'emulator-5554'
    scrcpy_windows_name = "LIO-AN00"
    训练数据截取(device_id, scrcpy_windows_name, True)