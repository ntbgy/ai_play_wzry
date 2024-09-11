import json
import os
import random
import shutil
import threading
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from pynput import keyboard
from pynput.keyboard import Key, Listener

from Batch import create_masks
from common import get_files_path
from common.env import 判断状态模型地址,状态词典B
from common.my_logger import logger
from resnet_utils import myResnet
from 模型_策略梯度 import Transformer

态 = '暂停'


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


def my_cv_imread(filepath):
    """
    支持路径有中文还能cv2读取
    """
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):
        return key.char
    else:

        return str(key)


# 监听按压
def on_press(key):
    global 态
    pass


# 监听释放
def on_release(key):
    global 态
    key_name = get_key_name(key)
    if key_name == 'Key.up':
        态 = '弃'
    elif key_name == 'Key.left':
        态 = '普通'
    elif key_name == 'Key.down':
        态 = '过'
    elif key_name == 'Key.right':
        态 = '死亡'
    elif key_name == 'a':
        态 = '击杀敌方英雄'
    elif key_name == 's':
        # 别太严格了，没补刀也算吧，毕竟没补刀也有经济，我方小兵或队友推塔也算吧
        态 = '击杀小兵或野怪或推掉塔'
    elif key_name == 'd':
        态 = '被击杀'
    elif key_name == 'w':
        态 = '被击塔攻击'
    if key == Key.esc:
        # 停止监听
        return False


# 开始监听
def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def 本地图片状态标注():
    global 态
    th = threading.Thread(target=start_listen, )
    th.daemon = True
    th.start()
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
    resnet101 = myResnet(mod)
    model_判断状态 = Transformer(6, 768, 2, 12, 0.0, 6 * 6 * 2048)
    model_判断状态.load_state_dict(torch.load(判断状态模型地址))
    model_判断状态.cuda(device)
    file_root_dir = r"E:\ai-play-wzry\训练数据样本\未用\1726028918"
    paths = get_files_path(file_root_dir)
    paths = [item for item in paths if '.jpg' in os.path.basename(item)]
    paths = random.sample(paths, 200)
    for index, image_path in enumerate(paths):
        image_name = os.path.basename(image_path)
        image_dir = os.path.basename(os.path.dirname(image_path))
        image_new_name = image_dir + '_' + image_name.replace('.jpg', '')
        image_new_path = f'E:\\ai-play-wzry\\判断数据样本\\{image_new_name}.jpg'
        if os.path.isfile(image_new_path):
            continue
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
        if 状况 in ['普通','死亡', '被击杀']:
            continue
        logger.info(状况)
        截图 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        截图 = cv2ImgAddText(截图, 状况, 0, 0, (000, 222, 111), 25)
        cv2.imshow('image', 截图)
        cv2.waitKey()
        while 态 == '暂停':
            time.sleep(0.02)
        新输出 = {}
        if 态 == '过':
            校准输出 = '过'
            continue
        elif 态 == '普通':
            校准输出 = '普通'
        elif 态 == '死亡':
            校准输出 = '死亡'
        elif 态 == '被击杀':
            校准输出 = '被击杀'
        elif 态 == '击杀小兵或野怪或推掉塔':
            校准输出 = '击杀小兵或野怪或推掉塔'
        elif 态 == '击杀敌方英雄':
            校准输出 = '击杀敌方英雄'
        elif 态 == '被击塔攻击':
            校准输出 = '被击塔攻击'
        elif 态 == '弃':
            态 = '暂停'
            continue
        else:
            logger.warning(f'{image_path}, {状况}')
            continue
        新输出 = {image_new_name: 校准输出}
        data_path = r'E:\ai-play-wzry\判断数据样本\判断新.json'
        if os.path.isfile(data_path):
            with open(data_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(新输出, ensure_ascii=False))
                f.write('\n')
        else:
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(新输出, ensure_ascii=False))
                f.write('\n')
        print(image_path, image_new_path)
        shutil.copy(image_path, image_new_path)
        print(状况, 校准输出)
        态 = '暂停'
        # image = my_cv_imread(path)
        # # 检查图像是否成功读取
        # if image is not None:
        #     # 显示图像
        #     cv2.imshow('Image', image)
        #     # 等待按键按下，0 表示无限等待
        #     cv2.waitKey(0)
        #     # 关闭所有窗口
        #     cv2.destroyAllWindows()
        # else:
        #     print("无法读取图像")
