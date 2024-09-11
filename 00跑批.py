import os
import time

from common.my_logger import logger

# os.system(
#     r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe C:\Users\ntbgy\PycharmProjects\ai-play-wzry\A3训练状态判断模型.py')
# time.sleep(5)

os.system(
    r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe C:\Users\ntbgy\PycharmProjects\ai-play-wzry\B2处理训练数据.py')
time.sleep(5)

os.system(
    r'C:\Users\ntbgy\.conda\envs\wzry38\python.exe C:\Users\ntbgy\PycharmProjects\ai-play-wzry\B3训练主模型.py')
logger.info('done')