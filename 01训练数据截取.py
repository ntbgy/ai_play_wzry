import logging
import threading

import torchvision
from airtest.cli.parser import cli_setup
from airtest.core.api import *
from pynput import keyboard
from pynput.keyboard import Key, Listener

from Batch import create_masks
from common.airtestProjectsCommon import get_now_img_txt, ocr_now_touch
from common.env import training_data_save_directory, device_id, scrcpy_windows_name
from common.my_logger import logger
from resnet_utils import myResnet
from 取训练数据 import *
from 杂项 import *
from 模型_策略梯度 import 智能体
from 辅助功能 import 状态信息综合
from 运行辅助 import *

# 设置日志级别
logger_airtest = logging.getLogger("airtest")
logger_ppocr = logging.getLogger("ppocr")
logger_airtest.setLevel(logging.CRITICAL)
logger_ppocr.setLevel(logging.CRITICAL)
# 连接设备
if not cli_setup():
    auto_setup(
        __file__,
        logdir=True,
        devices=["android:///"]
    )
# 获取当前文件绝对路径
dir_path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(training_data_save_directory):
    os.makedirs(training_data_save_directory)
lock = threading.Lock()
start = time.time()
fun_start = 0
time_interval = 0
数据字典 = {'interval_times': 0, 'max_interval': 0., 'interval_location': []}
index = 0
count = 0
count_dict = {'first_time': 0., 'first_p_to_second_r': 0.}
W键按下 = False
S键按下 = False
A键按下 = False
D键按下 = False
Q键按下 = False
手动模式 = False
AI打开 = True
操作列 = []
自动 = 0
智能体 = 智能体(
    动作数=7,
    并行条目数=100,
    学习率=0.0003,
    轮数=3,
    输入维度=6
)


def 单机模式返回大厅并重新开始1v1(txt=None):
    """不知道为什么后面调用，airtest的touch会点不了，是某个touch没有释放么？"""
    if txt is None:
        txt = get_now_img_txt()
    if '返回大厅' in txt:
        ocr_now_touch('返回大厅')
        sleep(5)
    if '开始练习' in txt:
        ocr_now_touch('1v1模式')
        sleep(1)
        ocr_now_touch('难度1')
        sleep(1)
        ocr_now_touch('开始练习')
        sleep(2)
    touch(Template(filename='auto/王者荣耀/对战/屏幕截图 2024-09-06 083209.png'))
    sleep(1)
    ocr_now_touch('射手')
    sleep(1)
    touch(Template(filename='auto/王者荣耀/对战/屏幕截图 2024-09-06 083341.png'))
    sleep(1)
    ocr_now_touch('挑选对手')
    sleep(1)
    touch(Template(filename='auto/王者荣耀/对战/屏幕截图 2024-09-06 083209.png'))
    sleep(1)
    ocr_now_touch('射手')
    sleep(1)
    touch(Template(filename='auto/王者荣耀/对战/屏幕截图 2024-09-06 083529.png'))
    sleep(1)
    ocr_now_touch('开始对战')
    sleep(3)


# txt = get_now_img_txt(dir_path)
# if '返回大厅' in txt or '开始练习' in txt:
#     try:
#         返回大厅并重新开始1v1(txt)
#     except TargetNotFoundError as e:
#         # 生成报告
#         simple_report(__file__, logpath=True, output=f"{dir_path}\\log\\log.html")
#         # 打开报告
#         os.startfile(f"{dir_path}\\log\\log.html")
#         raise TargetNotFoundError(e)


def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):
        return key.char
    else:
        return str(key)


# 监听按压
def on_press(key):
    global fun_start, time_interval, index, \
        数据字典, count, count_dict, W键按下, \
        S键按下, A键按下, D键按下, 手动模式, \
        操作列, AI打开, Q键按下
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
    elif key_name == 'i':
        AI打开 = bool(1 - AI打开)
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
    global start, fun_start, time_interval, index, count, count_dict, \
        W键按下, S键按下, A键按下, D键按下, \
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


# 批量跑就不执行监听程序了
th = threading.Thread(target=start_listen, )
th.start()  # 启动线程

词数词典路径 = "./json/词_数表.json"
数_词表路径 = "./json/数_词表.json"
操作查询路径 = "./json/名称_操作.json"
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
            if os.path.exists('stop_flag.txt'):
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
                logger.info((旧指令, '周期'))
                time.sleep(0.02)
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
                    if 操作词典['动作操作'] != '无动作' and 操作词典['动作操作'] != '发起集合' and 操作词典[
                        '动作操作'] != '发起进攻' and 操作词典['动作操作'] != '发起撤退':
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
        if os.path.exists('stop_flag.txt'):
            pyminitouch_device.stop()
            os.remove('stop_flag.txt')
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
