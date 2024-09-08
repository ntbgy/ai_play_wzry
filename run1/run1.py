import os
import threading

from 训练数据截取 import single_run

dir_path = os.path.dirname(os.path.abspath(__file__))
device_id = 'emulator-5554'
scrcpy_windows_name = "LIO-AN00"
flag_file_name = 'stop_flag_1.txt'
airtest_devices = "android://127.0.0.1:5037/emulator-5554"
def scrcpy():
    os.system('scrcpy -s emulator-5554 --max-size 960')
th1 = threading.Thread(target=scrcpy)
th1.start()
for _ in  range(24):
    th2 = threading.Thread(target=single_run, args=(
        dir_path, device_id, scrcpy_windows_name, flag_file_name, airtest_devices))
    th2.start()
    th2.join()