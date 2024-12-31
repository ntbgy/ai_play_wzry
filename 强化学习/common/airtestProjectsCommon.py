# from common.env import sleep_time
#
# print(f"sleep_time = {sleep_time}")
import os.path
import shutil
import sys

import cv2
import numpy as np
from airtest.core.api import *
from paddleocr import PaddleOCR

from common import get_dirs_path
from common.auto_game import get_now_img


# noinspection PyUnusedLocal
def ocr_now_touch(target_text, dir_path: any = None, show_result: bool = False, sleep_time: int = 1):
    res = ocr_touch(target_text, get_now_img(), show_result)
    if res:
        sleep(sleep_time)
    return res

# noinspection PyUnusedLocal
def get_now_img_txt(dir_path: any = None):
    return get_img_txt(get_now_img())


def ocr_touch(target_text, pic_path, show_result: bool = False):
    # 使用PaddleOCR识别图片文字
    ocr = PaddleOCR()
    ocr_result = ocr.ocr(img=pic_path, cls=True)
    # 遍历识别结果，找到目标文字的坐标
    target_coords = None
    txt = ''
    for line in ocr_result:
        for word_info in line:
            # 获取识别结果的文字信息
            textinfo = word_info[1][0]
            txt += textinfo + '\n'
            if target_text.lower() in textinfo.lower():
                # 获取文字的坐标（中心点）
                x1, y1 = word_info[0][0]
                x2, y2 = word_info[0][2]
                target_coords = (3.33 * (x1 + x2) / 2, 3.33 * (y1 + y2) / 2)
                break
        if target_coords:
            break

    # 使用Airtest点击坐标
    if target_coords:
        print(f'正在点击【{target_text}】，坐标{target_coords}')
        touch(target_coords)
        return True
    else:
        if show_result:
            print('#' * 50)
            print(f"未找到目标文字：{target_text}")
            print(f"识别文字：{txt}")
            print('#' * 50)
        return False


def execute(cmd):
    adb_str = "adb shell {}".format(cmd)
    print(adb_str)
    os.system(adb_str)


def swipe_example():
    # get the device width & height
    width, height = device().get_current_resolution()
    # cal swipe (start/end) point coordinate
    start_pt = (width * 0.9, height / 2)
    end_pt = (width * 0.1, height / 2)
    # swipe for 5 times:
    for i in range(5):
        swipe(start_pt, end_pt)
        sleep(1)  # wait for the device's response


def clean_log():
    # 清空日志
    if not os.path.isdir('./log'):
        return
    try:
        for name in os.listdir('./log'):
            os.remove(f'./log/{name}')
    except PermissionError as e:
        print(e)


def my_cv_imread(filepath):
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


def get_img_txt(pic_path) -> str:
    pic_path = str(pic_path)
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 可以根据需要选择语言
    # 使用PaddleOCR识别图片文字
    ocr_result = ocr.ocr(pic_path, cls=True)
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


def get_middle_coordinate(coordinates: list):
    import numpy as np

    coordinates = np.array(coordinates)

    # 计算每个维度的平均值
    average_x = np.mean(coordinates[:, 0])
    average_y = np.mean(coordinates[:, 1])

    # 中间坐标
    middle_coordinate = [average_x, average_y]
    return middle_coordinate


def clean_airtest_log(name='log'):
    paths = get_dirs_path(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
    )
    for path in paths:
        base_name = os.path.basename(path)
        if base_name == name:
            try:
                shutil.rmtree(path)
                print(f"目录 {path} 已成功删除")
            except Exception as e:
                print(f"删除目录时出错: {e}")


def os_chdir_modules(modules: str):
    os.chdir(
        os.path.dirname(
            os.path.abspath(sys.modules[modules].__file__)))


if __name__ == '__main__':
    clean_airtest_log('log')
    clean_airtest_log('temp')
