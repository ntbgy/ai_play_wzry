import ctypes
import logging
import subprocess
import time

import numpy as np
import pygetwindow as gw
# noinspection PyPackageRequirements
import win32gui
import win32ui
from PIL import Image
from paddleocr import PaddleOCR

"""
模拟器是雷电模拟器
模拟器分辨率设置为 3200*1440
投屏工具是scrcpy
投屏分辨率设置为 1600*720
scrcpy --window-title scrcpy投屏 --window-width=1600 --window-height=720
识图工具是paddleocr
获取窗口图片工具是win32gui
界面点击操作工具是adb
"""

logger_ppocr = logging.getLogger("ppocr")
logger_ppocr.setLevel(logging.ERROR)


class stopEnv:
    """
    这个类用于控制游戏的停止状态。

    属性:
        __stop (bool): 一个私有变量，指示游戏是否应该停止。

    方法:
        get_stop(): 获取当前的停止状态。
        set_stop(s): 设置停止状态。

    """

    def __init__(self):
        """
        初始化 stopEnv 类的实例。
        默认情况下，游戏状态为未停止。
        """
        # 初始化私有变量 __stop 为 False
        self.__stop = False

    def get_stop(self):
        """
        获取当前的停止状态。

        返回:
            bool: 如果游戏已停止，则返回 True，否则返回 False。
        """
        # 返回私有变量 __stop 的值
        return self.__stop

    def set_stop(self, s):
        """
        设置停止状态。

        参数:
            s (bool): 要设置的停止状态。True 表示停止游戏，False 表示继续游戏。

        返回:
            None
        """
        # 设置私有变量 __stop 的值为传入的参数 s
        self.__stop = s


# 创建 stopEnv 类的实例 se
se = stopEnv()


class ImageStorage:
    def __init__(self):
        """
        初始化 ImageStorage 类的实例。
        默认情况下，图像数据为空。
        """
        # 初始化私有变量 __image_data 为空
        self.__image_data = None
        self.__image_time = None

    def get_image_data(self):
        """
        获取当前存储的图像数据。

        返回:
            numpy.ndarray: 如果存储了图像数据，则返回图像数据，否则返回 None。
        """
        # 返回私有变量 __image_data 的值
        return self.__image_data, self.__image_time

    def set_image_data(self, image_np):
        """
        设置图像数据。

        参数:
            image_np (numpy.ndarray): 要设置的图像数据。

        返回:
            None
        """
        # 设置私有变量 __image_data 的值为传入的参数 image_np
        self.__image_data = image_np
        self.__image_time = time.time()

    def save_image(self, filename):
        """
        将存储的图像数据保存为文件。

        参数:
            filename (str): 要保存的文件名。

        返回:
            None
        """
        if self.__image_data is not None:
            # 将 numpy 数组转换为 PIL 图像
            image = Image.fromarray(self.__image_data)
            # 保存图像
            image.save(filename)
        else:
            print("No image data to save.")


# 创建 ImageStorage 类的实例 image_storage
image_storage = ImageStorage()


def get_window_image(hwnd):
    """窗口不能最小化哦！"""
    hwnd = win32gui.FindWindow(None, hwnd)
    # 获取整个窗口的位置和尺寸
    # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    # left, top, right, bottom = win32gui.GetClientRect(hwnd)
    # width = right - left
    # height = bottom - top
    width = 1600
    height = 720

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

    image.save(f'imgCatch/{time.time()}.png')

    # 将 PIL Image 对象转换为 numpy.ndarray 类型
    image_np = np.asarray(image)

    # 释放资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        image_storage.set_image_data(image_np)
        return image_np
    else:
        image_storage.set_image_data(None)
        return None


def get_now_img(hwnd='scrcpy投屏') -> any:
    now_time = time.time()
    img, img_time = image_storage.get_image_data()
    if img_time is None or now_time - img_time > 1:
        img = get_window_image(hwnd)
        return img
    else:
        return img


def get_img_txt(img: any) -> str:
    start_time = time.time()
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 可以根据需要选择语言
    # 使用PaddleOCR识别图片文字
    ocr_result = ocr.ocr(img, cls=True)
    txt = ''
    for line in ocr_result:
        if not line:
            return txt
        for word_info in line:
            # 获取识别结果的文字信息
            textinfo = word_info[1][0]
            txt += textinfo + '\n'
    txt = txt.strip()
    end_time = time.time()
    # 计算OCR识别耗时
    elapsed_time = end_time - start_time
    # 将耗时转换为小时、分钟和秒的格式
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    # 打印格式化后的耗时
    print(f"OCR识别耗时: {int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒")

    return txt


def check_stop():
    while True:
        """
        scrcpy --window-title scrcpy投屏 --window-width=1600 --window-height=720
        """
        time.sleep(30)
        txt = get_img_txt(get_now_img())
        print(txt)
        keywords = ['返回大厅', '再来一局', '继续', '胜利', '失败', '请选择', '皮肤', '开始匹配', '娱乐模式']
        if any(keyword in txt for keyword in keywords):
            print("检测到游戏结束")
            se.set_stop(True)
            break


def move_window_to_corner(window_title="scrcpy投屏"):
    try:
        # 尝试获取指定标题的窗口
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            window = windows[0]
            # 将窗口移动到左上角 (0, 0) 位置
            window.moveTo(0, 0)
            print(f"已将窗口 '{window_title}' 移动到左上角。")
        else:
            print(f"未找到标题为 '{window_title}' 的窗口。")
    except Exception as e:
        print(f"移动窗口时发生错误: {e}")


def adb_click(coordinate: tuple, sleep_time=None):
    x, y = coordinate
    if x < 1 and y < 1:
        x = int(x * 1600 * 2)
        y = int(y * 720 * 2)
    adb_command = f'adb shell input tap {x} {y}'
    print(adb_command)
    subprocess.run(adb_command, shell=True)
    if sleep_time is not None:
        time.sleep(sleep_time)


def ocr_adb_click(target_text, image, sleep_time=1, print_false_txt=False):
    # 使用PaddleOCR识别图片文字
    ocr = PaddleOCR()
    ocr_result = ocr.ocr(img=image, cls=True)
    # 遍历识别结果，找到目标文字的坐标
    target_coords = None
    txt = ''
    for line in ocr_result:
        if line is None:
            continue
        for word_info in line:
            # 获取识别结果的文字信息
            textinfo = word_info[1][0]
            txt += textinfo + '\n'
            if target_text in textinfo:
                # 获取文字的坐标（中心点）
                x1, y1 = word_info[0][0]
                x2, y2 = word_info[0][2]
                target_coords = ((x1 + x2) / 2, (y1 + y2) / 2)
                break
        if target_coords:
            break

    if target_coords is None:
        if print_false_txt is True:
            print('#' * 50)
            print(f"未找到目标文字：{target_text}")
            print(f"识别文字：{txt}")
            print('#' * 50)
        return False
    else:
        target_coords = (target_coords[0] * 2, target_coords[1] * 2)
        print(f"正在点击【{target_text}】，它的坐标：{target_coords}。")
        adb_click(target_coords, sleep_time)
        return True


def back_to_room():
    # 或许是胜负结果页面
    adb_click((0.5, 0.5), 3)
    # 点赞界面
    adb_click((0.5, 0.85), 5)
    # 本局表现
    adb_click((0.5, 0.85), 5)
    # 是否上分了
    adb_click((0.5, 0.85), 5)
    ocr_adb_click('确定', get_now_img(), 5)
    ocr_adb_click('返回房间', get_now_img(), 5)


def start_next_game():
    # 点击开始游戏
    ocr_adb_click('开始匹配', get_now_img(), 1)

    for _ in range(10):
        res = ocr_adb_click('确认', get_now_img(), 1)
        if res is True:
            break

    for _ in range(10):
        res = ocr_adb_click('全部', get_now_img(), 1)
        if res is True:
            break

    # 展开
    adb_click((346 * 2, 340 * 2), 3)
    ocr_adb_click('游走', get_now_img(), 2)

    # 点击瑶
    # adb_click((978*2, 397*2), 3)
    adb_click((616 * 2, 397 * 2), 3)

    for _ in range(10):
        res = ocr_adb_click('确定', get_now_img(), 1)
        if res is False:
            time.sleep(4)
            break


def 钻石夺宝循环单抽(num=1):
    def 钻石夺宝单抽连抽():
        ocr_adb_click("买1", get_now_img(), 1)
        adb_click((0.5, 0.5), 1)
        adb_click((0.5, 0.5), 1)
        ocr_adb_click('确定', get_now_img(), 1)
        ocr_adb_click('确定', get_now_img(), 1)

    for _ in range(num):
        钻石夺宝单抽连抽()


if __name__ == '__main__':
    start_next_game()
    # back_to_room()
