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


# 定义文件路径
file_path = r"E:\ai-play-wzry\训练数据样本\未用\1725970688\_操作数据.json"

# 打开文件，读取内容
with open(file_path, 'r', encoding='ansi') as f:
    txt = f.read()

# 遍历文件中的每一行
for line in txt.split('\n'):
    # 将每一行转换为字典
    line = eval(line)
    # 构建图片的完整路径
    image = os.path.dirname(file_path) + f'\\{line["图片号"]}.jpg'
    # 使用自定义函数读取图片
    image = my_cv_imread(image)
    # 打印图片信息
    print(json.dumps(line, ensure_ascii=False, indent=2))
    # 显示图片
    cv2.imshow('image', image)
    # 等待按键
    cv2.waitKey()
    # 清屏
    os.system('cls')
