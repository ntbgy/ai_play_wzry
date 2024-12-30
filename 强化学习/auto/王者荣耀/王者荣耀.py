"""
com.tencent.tmgp.sgame
"""
import logging
import sys

from airtest.cli.parser import cli_setup
from airtest.core.api import *
from airtest.report.report import simple_report

from auto.王者荣耀.每日领垃圾.领垃圾 import 领垃圾
from auto.王者荣耀.进入游戏主页.进入游戏主页 import 进入游戏主页
from common.airtestProjectsCommon import clean_log


def 退出王者荣耀():
    stop_app("com.tencent.tmgp.sgame")
    sleep(2)
    home()
    sleep(1)
    home()
    sleep(1)


def 王者荣耀(dir_path):
    # 进入王者荣耀首页
    os.chdir(
        os.path.dirname(os.path.abspath(sys.modules['auto.王者荣耀.进入游戏主页.进入游戏主页'].__file__)))
    进入游戏主页()
    # 执行自动化脚本
    os.chdir(os.path.dirname(os.path.abspath(sys.modules['auto.王者荣耀.每日领垃圾.领垃圾'].__file__)))
    领垃圾(dir_path)
    # os.chdir(
    #     os.path.dirname(os.path.abspath(sys.modules['auto.王者荣耀.夫子的试炼.夫子的进阶试验'].__file__)))
    # 夫子的进阶试验(dir_path)
    # 退出王者荣耀
    # 退出王者荣耀()


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
    # adb = ADB(serialno="emulator-5554")
    # recorder = Recorder(adb)
    # 开启录屏
    # 需要先安装adb，可通过Android studio安装，或许要手工添加环境变量
    # recorder.start_recording(max_time=20)
    # 执行自动化程序
    王者荣耀(dir_path)
    # 退出王者荣耀()
    # 生成报告
    simple_report(__file__, logpath=True, output=f"{dir_path}\\log\\log.html")
    # 打开报告
    # os.startfile(f"{dir_path}\\log\\log.html")
    # 结束录屏
    # recorder.stop_recording(output="test.mp4")
