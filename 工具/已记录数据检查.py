import json
import os.path

import cv2
import numpy as np


def my_cv_imread(filepath):
    """
    支持路径有中文还能cv2读取
    """
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


file_path = r"E:\ai-play-wzry\训练数据样本\未用\1725970688\_操作数据.json"
with open(file_path, 'r', encoding='ansi') as f:
    txt = f.read()
for line in txt.split('\n'):
    line = eval(line)
    image = os.path.dirname(file_path) + f'\\{line["图片号"]}.jpg'
    image = my_cv_imread(image)
    print(json.dumps(line, ensure_ascii=False, indent=2))
    cv2.imshow('image', image)
    cv2.waitKey()
    # pycharm 执行不会清屏，可 CMD 执行
    os.system('cls')
