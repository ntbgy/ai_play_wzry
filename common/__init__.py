import ctypes
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import win32gui
import win32ui
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from pynput import keyboard
from torch.autograd import Variable

from common.Batch import create_masks
from common.Transformer import Transformer
from common.env import 判断状态模型地址, 状态词典B, 状态词典
from common.resnet_utils import myResnet


def get_dirs_path(dir_path):
    paths = list()
    for root, directory, files in os.walk(dir_path):
        paths.append(Path(root))
    return paths


def get_files_path(dir_path):
    paths = list()
    for root, directory, files in os.walk(dir_path):
        for file in files:
            paths.append(Path(root) / Path(file))
    return paths


def get_txt(pic: any):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 可以根据需要选择语言

    # 使用PaddleOCR识别图片文字
    ocr_result = ocr.ocr(pic, cls=True)

    txt = ''
    for line in ocr_result:
        if not line:
            return txt
        for word_info in line:
            # 获取识别结果的文字信息
            textinfo = word_info[1][0]
            txt += textinfo + '\n'
    txt = txt.strip()
    return txt


def 读取训练数据(路径):
    输入表单 = []
    输出表单 = []
    with open(路径, encoding='utf-8') as f:
        while True:
            行 = f.readline()
            if not 行:
                break
            json_行 = json.loads(行)

            内容 = json_行['内容______']
            内容_输入 = 内容['输入______']
            内容_输出 = 内容['输出______']
            # 这里的数据还得进行分割先暂时分割成16份吧
            单元长度 = len(内容_输入) // 16
            for i in range(16):
                # print(内容_输入[i*单元长度:(i+1)*单元长度])
                输入表单.append(内容_输入[i * 单元长度:(i + 1) * 单元长度])
                输出表单.append(内容_输出[i * 单元长度:(i + 1) * 单元长度])
    return 输入表单, 输出表单


def 写出词标号引索(总词表, 词_数表路径, 数_词表路径):
    print("正在写出词的标号引索数据可能需要较长时间")
    标号_到_字符 = {}
    字符_到_标号 = {}
    标号_字符 = []

    # 标号_到_字符 = list(set(总表单))
    i = 0
    j = 0
    for 词表 in 总词表:
        j = j + 1
        for 字符 in 词表:

            if 字符 not in 标号_字符:
                标号_字符.append(字符)
                字符_到_标号[字符] = i
                标号_到_字符[i] = 字符
                i = i + 1
        if j % 10000 == 0:
            print(i, 标号_到_字符[i - 1], j / len(总词表))

    # print(标号_到_字符[1], 标号_到_字符[111], len(标号_到_字符))
    with open(词_数表路径, 'w', encoding='utf-8') as f:
        json.dump(字符_到_标号, f, ensure_ascii=False)
    with open(数_词表路径, 'w', encoding='utf-8') as f:
        json.dump(标号_到_字符, f, ensure_ascii=False)


def 生成训练用numpy数组(输入表单, 词_数表, numpy数组路径):
    表_1 = []

    表_2 = []

    i = 0
    临 = ''
    for 表单 in 输入表单:
        表_3 = []
        for 字符 in 表单:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':

                    临 = 字符
                else:
                    临 = 临 + 字符
            else:

                if 临 == '':

                    if 字符.lower() in 词_数表:

                        表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(14999)
                else:
                    if 临.lower() in 词_数表:

                        表_3.append(词_数表[临.lower()])
                    else:
                        表_3.append(14999)
                    临 = ''
                    if 字符.lower() in 词_数表:

                        表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(14999)
        if 临 != '':
            if 临.lower() in 词_数表:

                表_3.append(词_数表[临.lower()])
            else:
                表_3.append(14999)
            临 = ''

        if len(表_3) != 667:
            # 表_1.append(np.array(表_3[0:-1]))
            # 表_2.append(np.array(表_3[1:]))
            print(表_3)
        else:

            表_1.append(np.array(表_3[0:-1]))
            表_2.append(np.array(表_3[1:]))
        if i % 1000 == 0:
            print("数据转化为numpy数组完成度百分比{}".format(i / len(输入表单) * 100))
        i = i + 1
    print("数据转化为numpy数组完成。")

    输入np = np.array(表_1)
    输出np = np.array(表_2)
    np.savez(numpy数组路径, 输出np=输出np, 输入np=输入np)


def 生成测试用numpy数组(输入表单, 词_数表):
    表_1 = []

    for 字符 in 输入表单:
        if 字符.lower() in 词_数表:
            表_1.append(词_数表[字符])
        else:
            表_1.append(14999)
    输入np = np.array(表_1)
    return (输入np)


def 生成训练用numpy数组_A(输入表单, 词_数表, numpy数组路径):
    表_1 = []

    表_2 = []

    i = 0
    临 = ''
    for 表单 in 输入表单:
        表_3 = []
        for 字符 in 表单:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':

                    临 = 字符
                else:
                    临 = 临 + 字符
            else:

                if 临 == '':

                    if 字符.lower() in 词_数表:
                        if 字符 != ' ':
                            表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(14999)
                else:
                    if 临.lower() in 词_数表:
                        if 临 != ' ':
                            表_3.append(词_数表[临.lower()])
                    else:
                        表_3.append(14999)
                    临 = ''
                    if 字符.lower() in 词_数表:
                        if 字符 != ' ':
                            表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(14999)
        if 临 != '':
            if 临.lower() in 词_数表:
                if 字符 != ' ':
                    表_3.append(词_数表[临.lower()])
            else:
                表_3.append(14999)
            临 = ''

        if len(表_3) != 667:
            # 表_1.append(np.array(表_3[0:-1]))
            # 表_2.append(np.array(表_3[1:]))
            print(表_3)
        else:

            表_1.append(np.array(表_3[0:-1]))
            表_2.append(np.array(表_3[1:]))
        if i % 1000 == 0:
            print("数据转化为numpy数组完成度百分比{}".format(i / len(输入表单) * 100))
        i = i + 1
    print("数据转化为numpy数组完成。")

    输入np = np.array(表_1)
    输出np = np.array(表_2)
    np.savez(numpy数组路径, 输出np=输出np, 输入np=输入np)


def 读取训练数据_A(路径):
    输入表单 = []
    with open(路径, encoding='utf-8') as f:
        while True:
            行 = f.readline()
            if not 行:
                break
            json_行 = json.loads(行)

            内容 = json_行['input']
            输入表单.append(内容)

    return 输入表单


def 生成测试用numpy数组_A(输入表单, 词_数表):
    表_3 = []
    临 = ''

    for 字符 in 输入表单:
        if 字符.lower() in 词_数表:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':

                    临 = 字符
                else:
                    临 = 临 + 字符
            else:

                if 临 == '':

                    if 字符.lower() in 词_数表:
                        if 字符.lower() != ' ':
                            表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(14999)
                else:
                    if 临.lower() in 词_数表:
                        if 临.lower() != ' ':
                            表_3.append(词_数表[临.lower()])
                    else:
                        表_3.append(14999)
                    临 = ''
                    if 字符.lower() in 词_数表:
                        if 字符.lower() != ' ':
                            表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(14999)
    输入np = np.array(表_3)
    return (输入np)


def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    np_mask = np_mask.cuda(device)
    return np_mask


def 打印测试数据(数_词表, 数据, 输人_分, 标签):
    临 = 数据[0]
    欲打印 = [数_词表[str(临[i])] for i in range(临.size)]
    打印 = ""
    for i in range(len(欲打印)):
        打印 = 打印 + 欲打印[i]

    临 = 输人_分.cpu().numpy()[0]
    欲打印2 = [数_词表[str(临[i])] for i in range(输人_分.size(1))]
    # 欲打印2=str(欲打印2)
    # print("输入：", 欲打印2)
    if 标签 == 打印:
        return True
    else:
        print(打印)
        return False


def 打印测试数据_A(数_词表, 数据, 输人_分):
    if 数据.shape[0] != 0:

        临 = 数据[0]
        欲打印 = [数_词表[str(临[i])] for i in range(临.size)]
        打印 = ""
        for i in range(len(欲打印)):
            打印 = 打印 + 欲打印[i]

        临 = 输人_分.cpu().numpy()[0]
        欲打印2 = [数_词表[str(临[i])] for i in range(输人_分.size(1))]
        欲打印2 = str(欲打印2)
        # print("输入：", 欲打印2)
        print("输出：", 打印)


def get_window_image(hwnd):
    """窗口不能最小化哦！"""
    hwnd = win32gui.FindWindow(None, hwnd)
    # 获取整个窗口的位置和尺寸
    # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top

    # 获取窗口的设备上下文
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # 创建位图对象
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)

    # 将位图选入内存设备上下文
    saveDC.SelectObject(saveBitMap)

    # 将窗口图像复制到内存设备上下文
    result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)

    # 获取位图信息
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    # 构建 PIL Image 对象
    image = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    # 释放资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        return image
    else:
        return None


def 打印抽样数据(数_词表, 数据, 输出_分):
    临 = 数据[0]
    欲打印 = [数_词表[str(临[i, 0])] for i in range(0, 临.shape[0])]
    临 = 输出_分.cpu().numpy()
    欲打印2 = [数_词表[str(临[i])] for i in range(0, 临.shape[0])]
    print("抽样输出", 欲打印)
    print("目标输出", 欲打印2)


def 读出引索(词_数表路径, 数_词表路径):
    with open(词_数表路径, encoding='utf-8') as f:
        词_数表 = json.load(f)

    with open(数_词表路径, encoding='utf-8') as f:
        数_词表 = json.load(f)
    return 词_数表, 数_词表


def 状态信息综合(图片张量, 操作序列, trg_mask):
    状态 = {'图片张量': 图片张量[np.newaxis, :], '操作序列': 操作序列, 'trg_mask': trg_mask}
    return 状态


def my_cv_imread(filepath):
    """
    支持路径有中文还能cv2读取
    """
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


def get_key_name(key):
    """从pynput.keyboard模块中的按键对象中提取出一个可识别的键名"""
    if isinstance(key, keyboard.KeyCode):
        return key.char
    else:

        return str(key)


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        'C:/Windows/Fonts/STHUPO.TTF', textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def get_now():
    current_time = time.time()
    millisecond_part = str(int((current_time % 1) * 100)).zfill(2)
    time_struct = time.localtime(current_time)
    formatted_time = time.strftime('%Y%m%d%H%M%S', time_struct)
    return f"{formatted_time}{millisecond_part}"


def get_state_score(image_path):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
    resnet101 = myResnet(mod)
    model_判断状态 = Transformer(6, 768, 2, 12, 0.0, 6 * 6 * 2048)
    model_判断状态.load_state_dict(torch.load(判断状态模型地址))
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
    state = 状态列表[抽样np[0, 0, 0, 0]]
    score = 状态词典[state]
    return state, score


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def get_model(opt, trg_vocab, model_weights='model_weights'):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None and os.path.isfile(opt.load_weights + '/' + model_weights):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/{model_weights}'))
    else:
        量 = 0
        for p in model.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                a = 0
            长 = len(p.shape)
            点数 = 1
            for j in range(长):
                点数 = p.shape[j] * 点数

            量 += 点数
        print('使用参数:{}百万'.format(量 / 1000000))
    return model


def 处理状态参数(状态组, device):
    最长 = 0
    状态组合 = {}

    # 操作序列 = np.ones((1,))
    for 状态A in 状态组:
        if 状态A['图片张量'].shape[1] > 最长:
            最长 = 状态A['图片张量'].shape[1]
    for 状态 in 状态组:
        状态A = 状态.copy()
        if 状态A['图片张量'].shape[1] == 最长:
            单元 = 状态A
            操作序列 = np.ones((最长,))
            遮罩序列 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device).unsqueeze(0)
            单元['遮罩序列'] = 遮罩序列

        else:
            有效长度 = 状态A['图片张量'].shape[1]
            差值 = 最长 - 有效长度
            形状 = 状态A['图片张量'].shape
            图片张量_拼接 = torch.zeros(形状[0], 差值, 形状[2], 形状[3]).cuda(device).float()
            图片张量_拼接 = 图片张量_拼接.cpu().numpy()
            状态A['图片张量'] = np.append(状态A['图片张量'], 图片张量_拼接, axis=1)
            # 状态A['图片张量'] = torch.cat((状态A['图片张量'], 图片张量_拼接), 1)
            形状 = 状态A['角度集张量_序列'].shape
            角度集张量_拼接 = torch.zeros(形状[0], 差值, 形状[2]).cuda(device).float()
            状态A['角度集张量_序列'] = torch.cat((状态A['角度集张量_序列'], 角度集张量_拼接), 1)

            形状 = 状态A['位置张量_序列'].shape
            位置张量_拼接 = torch.zeros(形状[0], 差值, 形状[2]).cuda(device).float()
            状态A['位置张量_序列'] = torch.cat((状态A['位置张量_序列'], 位置张量_拼接), 1)

            形状 = 状态A['速度张量_序列'].shape
            速度张量_拼接 = torch.zeros(形状[0], 差值, 形状[2]).cuda(device).float()
            状态A['速度张量_序列'] = torch.cat((状态A['速度张量_序列'], 速度张量_拼接), 1)

            操作序列 = np.ones((有效长度,))
            遮罩序列 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device).unsqueeze(0)
            状态A['遮罩序列'] = 遮罩序列
            操作序列 = np.ones((差值,)) * -1
            遮罩序列 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device).unsqueeze(0)
            状态A['遮罩序列'] = torch.cat((状态A['遮罩序列'], 遮罩序列), 1)
            单元 = 状态A

        if 状态组合 == {}:
            状态组合 = 单元
        else:
            状态组合['遮罩序列'] = torch.cat((状态组合['遮罩序列'], 单元['遮罩序列']), 0)
            状态组合['速度张量_序列'] = torch.cat((状态组合['速度张量_序列'], 单元['速度张量_序列'],), 0)
            状态组合['位置张量_序列'] = torch.cat((状态组合['位置张量_序列'], 单元['位置张量_序列']), 0)
            状态组合['角度集张量_序列'] = torch.cat((状态组合['角度集张量_序列'], 单元['角度集张量_序列']), 0)
            # 状态组合['图片张量'] = torch.cat((状态组合['图片张量'], 单元['图片张量']), 0)
            状态组合['图片张量'] = np.append(状态组合['图片张量'], 单元['图片张量'], axis=0)
    src_mask, trg_mask = create_masks(状态组合['遮罩序列'], 状态组合['遮罩序列'], device)
    状态组合['trg_mask'] = trg_mask
    return 状态组合
