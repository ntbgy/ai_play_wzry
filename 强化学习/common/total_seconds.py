import time

from common.my_logger import logger


def total_seconds(time_start):
    time_end = time.time()
    total_seconds = time_end - time_start
    # 计算小时数
    hours = int(total_seconds // 3600)
    # 计算分钟数
    minutes = int((total_seconds % 3600) // 60)
    # 计算秒数
    seconds = int(total_seconds % 60)
    logger.debug(f"用时: {hours}小时 {minutes}分钟 {seconds}秒")
