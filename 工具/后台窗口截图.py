import ctypes

import win32gui
import win32ui
from PIL import Image


def _get_window_image(hwnd):
    """窗口不能最小化哦！"""
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


if __name__ == '__main__':
    # 获取窗口句柄（根据实际情况修改窗口标题）
    hwnd = win32gui.FindWindow(None, "LIO-AN00")
    image = _get_window_image(hwnd)
    image.save('window_screenshot.png')
