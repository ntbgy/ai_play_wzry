import logging
import os
from pathlib import Path

from paddleocr import PaddleOCR

from env import *
from 运行辅助 import get_window_image

# 获取当前文件绝对路径
dir_path = os.path.dirname(os.path.abspath(__file__))
# 设置日志级别
logger_airtest = logging.getLogger("airtest")
logger_ppocr = logging.getLogger("ppocr")
logger_airtest.setLevel(logging.CRITICAL)
logger_ppocr.setLevel(logging.CRITICAL)


def frend_player_check_for_ocr_txt(英雄='后羿', 玩家='后羿'):
    """不行呢~"""
    while True:
        image = get_window_image(scrcpy_windows_name)
        if not os.path.isdir('log'):
            os.mkdir('log')
        pic_path = str(Path(dir_path) / Path('log/window_screenshot.png'))
        image.save(pic_path)
        # 初始化PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 可以根据需要选择语言
        # 使用PaddleOCR识别图片文字
        ocr_result = ocr.ocr(pic_path, cls=True)
        for line in ocr_result:
            if not line:
                break
            for word_info in line:
                # 获取识别结果的文字信息
                textinfo = word_info[1][0]
                print(textinfo)
                if 玩家 in textinfo:
                    print(word_info)
        break


def frend_player_check_for_yolo(英雄='后羿', 玩家='后羿'):
    pass


def 向目标位置移动():
    pass


def 贴贴():
    pass


def main():
    pass


if __name__ == '__main__':
    frend_player_check_for_yolo()
