import logging
import sys

from airtest.cli.parser import cli_setup
from airtest.core.api import *
from airtest.report.report import simple_report

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
        if i==9:
            raise ValueError("进不去了")
def 结束对战回到首页():
    pass
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
    进入对战()