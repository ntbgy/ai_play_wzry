# -*- encoding=utf8 -*-
__author__ = "AirtestProject"

# crop_image()方法在airtest.aircv中，需要引入
from airtest.aircv import *
from airtest.cli.parser import cli_setup
from airtest.core.api import *

if not cli_setup():
    auto_setup(
        __file__,
        logdir=True,
        devices=["Android:///", ],
        # project_root="C:/Users/ntbgy/PycharmProjects/python3Project/airtestProjects/夫子的试炼"
    )

auto_setup(__file__)
screen = G.DEVICE.snapshot()
# 局部截图
screen = aircv.crop_image(screen, (1084, 258, 2660, 1124))
# 保存局部截图到log文件夹中
filename = try_log_screen(screen)['screen']
print(filename)
