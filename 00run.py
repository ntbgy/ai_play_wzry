import logging
import subprocess
import threading
from pipes import Template

from airtest.cli.parser import cli_setup
from airtest.core.api import *

from auto.王者荣耀.对战.对战 import 离线5v5
from common.airtestProjectsCommon import clean_log, os_chdir_modules
from common.my_logger import logger

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

# 模拟启动游戏的脚本
def start_game():
    logger.info('start_game')
    os_chdir_modules('auto.王者荣耀.对战.对战')
    离线5v5(dir_path)
    os.chdir(dir_path)

# 模拟检测游戏是否结束的程序
def check_game_status(process):
    game_running = True
    while game_running:
        time.sleep(2 * 60)
        if (not exists(Template(filename='攻击1.png'))) and (not exists(Template(filename='攻击2.png'))):
            game_running = False
            logger.info("检测到游戏结束")
    if process.poll() is None:
        process.terminate()

def run():
    for i in range(35):
        logger.info(f'第{i+1}局游戏开始！')
        start_game()
        script_path = os.path.join(dir_path, '01训练数据截取.py')
        process = subprocess.Popen([r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe', script_path])
        stop_event = threading.Event()
        time.sleep(6 * 60)
        check_thread = threading.Thread(target=check_game_status, args=(process,))
        check_thread.start()
        check_thread.join()
        logger.info("结束 AI 线程")
        time.sleep(5)
        logger.info(f'第{i + 1}局游戏结束！')

    os.system(r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe 02处理训练数据.py')
    os.system(r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe 03训练主模型.py')
    logger.info('done')

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        logger.warning("用户中断了程序的运行")