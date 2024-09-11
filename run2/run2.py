import logging
import os
import threading

from airtest.cli.parser import cli_setup
from airtest.core.api import auto_setup

from B1训练数据截取 import single_run
from common.airtestProjectsCommon import clean_log

# 设置日志级别
logger_airtest = logging.getLogger("airtest")
logger_ppocr = logging.getLogger("ppocr")
logger_ppocr.setLevel(logging.ERROR)
logger_airtest.setLevel(logging.ERROR)
# 清空存量日志
clean_log()
dir_path = os.path.dirname(os.path.abspath(__file__))
device_id = 'emulator-5556'
scrcpy_windows_name = "GM1900"
flag_file_name = 'stop_flag_2.txt'
airtest_devices = "android://127.0.0.1:5037/emulator-5556"
# 连接设备
if not cli_setup():
    auto_setup(
        __file__,
        logdir=True,
        devices=[airtest_devices]
    )
def scrcpy():
    os.system('scrcpy -s emulator-5556 --max-size 960')
th1 = threading.Thread(target=scrcpy)
th1.start()

for _ in  range(24):
    try:
        single_run(dir_path, device_id, scrcpy_windows_name, flag_file_name)
    except:
        continue
