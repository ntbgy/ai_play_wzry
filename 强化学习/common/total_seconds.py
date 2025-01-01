import time
import unittest

from common.my_logger import logger


def total_seconds(time_start):
    """
    计算从给定的起始时间到当前时间的总秒数，并输出到日志中。

    参数:
        time_start (float): 起始时间，以秒为单位，从 epoch 时间开始计算。

    返回:
        None
    """
    # 获取当前时间
    time_end = time.time()
    # 计算时间差
    total_seconds = time_end - time_start
    # 计算小时数
    hours = int(total_seconds // 3600)
    # 计算分钟数
    minutes = int((total_seconds % 3600) // 60)
    # 计算秒数
    seconds = float(total_seconds % 60)
    # 输出到日志
    logger.debug(f"用时: {hours}小时 {minutes}分钟 {seconds}秒")


class TestTotalSeconds(unittest.TestCase):
    """
    测试 total_seconds 函数的单元测试类。
    """

    def test_total_seconds(self):
        """
        测试 total_seconds 函数的各种情况。
        """
        # 测试正常情况
        time_start = time.time() - 3661  # 模拟过去 1 小时 1 分钟 1 秒
        total_seconds(time_start)

        # 测试边界情况，时间差为 0
        time_start = time.time()
        total_seconds(time_start)

        # 测试时间差非常小的情况
        time_start = time.time() - 0.001
        total_seconds(time_start)


if __name__ == '__main__':
    unittest.main()
