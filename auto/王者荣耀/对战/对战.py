import logging

from airtest.cli.parser import cli_setup

from common.airtestProjectsCommon import *


def 进入对战():
    ocr_now_touch('对战')
    sleep(1)
    ocr_now_touch('王者峡谷')
    sleep(1)
    ocr_now_touch('人机')
    sleep(1)
    ocr_now_touch('难度1')
    sleep(1)
    ocr_now_touch('开始练习')
    sleep(1)
    ocr_now_touch('开始匹配')
    sleep(1)
    for i in range(10):
        txt = get_now_img_txt()
        if '确认' in txt:
            ocr_now_touch('确认')
            sleep(1)
        else:
            sleep(5)
        if i == 9:
            raise ValueError("进不去了")


def 结束对战回到首页():
    pass


def 离线1v1():
    # 需要先手工登录一个账号，退出了不行~
    txt = get_now_img_txt()
    if '开始游戏' not in txt:
        stop_app("com.tencent.tmgp.sgame")
        sleep(1.0)
        home()
        sleep(1.0)
        start_app("com.tencent.tmgp.sgame")
        sleep(20)
    touch(Template(filename='屏幕截图 2024-09-06 082752.png'))
    sleep(1)
    ocr_now_touch('开始练习')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083209.png'))
    sleep(1)
    ocr_now_touch('射手')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083341.png'))
    sleep(1)
    ocr_now_touch('挑选对手')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083209.png'))
    sleep(1)
    ocr_now_touch('射手')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083529.png'))
    sleep(1)
    ocr_now_touch('开始对战')
    sleep(5)
    return '开始对战'


if __name__ == '__main__':
    # 设置日志级别
    logger_airtest = logging.getLogger("airtest")
    logger_ppocr = logging.getLogger("ppocr")
    logger_airtest.setLevel(logging.ERROR)
    logger_ppocr.setLevel(logging.ERROR)
    # 清空存量日志
    clean_log()
    # 连接设备
    if not cli_setup():
        auto_setup(
            __file__,
            logdir=True,
            devices=["android:///"]
        )
    # 获取当前文件绝对路径
    dir_path = os.path.dirname(os.path.abspath(__file__))
    离线1v1()
