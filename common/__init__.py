import os
from pathlib import Path

from paddleocr import PaddleOCR


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