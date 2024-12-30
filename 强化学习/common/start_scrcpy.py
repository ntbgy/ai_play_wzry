# 打开scrcpy
import os


def start_scrcpy(s='--max-size 960 --always-on-top'):
    """
    启动scrcpy

    参数:
        s (str): 启动参数，默认为 '--max-size 960 --always-on-top'

    返回:
        None
    """
    # 使用os.system函数执行scrcpy命令，启动scrcpy
    os.system(f'scrcpy {s}')
