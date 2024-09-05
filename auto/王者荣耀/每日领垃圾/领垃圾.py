import logging
from pathlib import Path

from airtest.cli.parser import cli_setup
from airtest.core.api import *
from airtest.report.report import simple_report

from common.airtestProjectsCommon import get_img_txt, clean_log


def my_home():
    for _ in range(3):
        home()
        sleep(0.5)


def confirm():
    for _ in range(5):
        if exists(Template(r"tpl1723362439365.png", record_pos=(0.001, 0.146), resolution=(3200, 1440))):
            touch(Template(r"tpl1723362439365.png", record_pos=(0.001, 0.146), resolution=(3200, 1440)))
        if exists(Template(r"tpl1723364714552.png", record_pos=(-0.0, 0.094), resolution=(3200, 1440))):
            touch(Template(r"tpl1723364714552.png", record_pos=(-0.0, 0.094), resolution=(3200, 1440)))
        if exists(Template(r"tpl1723362476177.png", record_pos=(-0.002, 0.143), resolution=(3200, 1440))):
            touch(Template(r"tpl1723362476177.png", record_pos=(-0.002, 0.143), resolution=(3200, 1440)))
        if exists(Template(r"tpl1723362241684.png", record_pos=(0.345, -0.163), resolution=(3200, 1440))):
            touch(Template(r"tpl1723362241684.png", record_pos=(0.345, -0.163), resolution=(3200, 1440)))
        if exists(Template(r"tpl1723363813367.png", record_pos=(0.003, 0.193), resolution=(3200, 1440))):
            touch(Template(r"tpl1723363813367.png", record_pos=(0.003, 0.193), resolution=(3200, 1440)))
            for _ in range(3):
                touch((1629, 775))
            if exists(Template(r"tpl1723362961788.png", record_pos=(-0.083, 0.155), resolution=(3200, 1440))):
                touch(Template(r"tpl1723362961788.png", record_pos=(-0.083, 0.155), resolution=(3200, 1440)))
        if exists(Template(r"tpl1723365009853.png", record_pos=(-0.402, -0.201), resolution=(3200, 1440))):
            return


def yi_jian_ling_qu():
    if exists(Template(r"tpl1723363700031.png", rgb=True, record_pos=(0.362, 0.182), resolution=(3200, 1440))):
        touch(Template(r"tpl1723363700031.png", rgb=True, record_pos=(0.362, 0.182), resolution=(3200, 1440)))
    else:
        return
    confirm()


def 领垃圾(dir_path):
    # 小妲己一键领取
    touch((2932, 1257))
    sleep(1.0)
    if exists(Template(r"tpl1723362371269.png", threshold=0.8, record_pos=(-0.055, 0.087), resolution=(3200, 1440))):
        touch(Template(r"tpl1723362371269.png", record_pos=(-0.055, 0.087), resolution=(3200, 1440)))
        sleep(1.0)
        pic_path = Path(dir_path) / 'log/now.png'
        snapshot(filename='now.png')
        txt = get_img_txt(pic_path)
        if "能力测试奖励" in txt:
            touch(Template(r"tpl1723405198078.png", record_pos=(0.274, -0.163), resolution=(3200, 1440)))
        if exists(Template(r"tpl1723403368116.png", record_pos=(0.403, -0.174), resolution=(3200, 1440))):
            touch(Template(r"tpl1723367630863.png", record_pos=(-0.41, -0.203), resolution=(3200, 1440)))
        else:
            confirm()
    sleep(1.0)
    touch(Template(r"tpl1723362540925.png", record_pos=(-0.36, -0.203), resolution=(3200, 1440)))
    sleep(3)
    # 邮件领取
    touch(Template(r"tpl1723362859045.png", record_pos=(0.329, -0.202), resolution=(3200, 1440)))
    sleep(1.0)
    # 好友邮件
    touch(Template(r"tpl1723362916763.png", record_pos=(-0.406, -0.15), resolution=(3200, 1440)))
    # 赠送金币
    touch(Template(r"tpl1723362923821.png", record_pos=(-0.188, -0.16), resolution=(3200, 1440)))
    if exists(Template(r"tpl1723362931216.png", record_pos=(0.36, 0.186), resolution=(3200, 1440))):
        touch(Template(r"tpl1723362931216.png", record_pos=(0.36, 0.186), resolution=(3200, 1440)))
        sleep(1.0)
        if exists(Template(r"tpl1723362476177.png", record_pos=(-0.002, 0.143), resolution=(3200, 1440))):
            touch(Template(r"tpl1723362476177.png", record_pos=(-0.002, 0.143), resolution=(3200, 1440)))
    # 收到邮件
    touch(Template(r"tpl1723363581948.png", record_pos=(-0.295, -0.162), resolution=(3200, 1440)))
    if exists(Template(r"tpl1723365117858.png", record_pos=(0.357, 0.185), resolution=(3200, 1440))):
        touch(Template(r"tpl1723365117858.png", record_pos=(0.357, 0.185), resolution=(3200, 1440)))
        sleep(1.0)
        confirm()
    # 系统邮件
    touch(Template(r"tpl1723363617489.png", record_pos=(-0.406, -0.102), resolution=(3200, 1440)))
    yi_jian_ling_qu()
    touch(Template(r"tpl1723365009853.png", record_pos=(-0.402, -0.201), resolution=(3200, 1440)))
    touch(Template(r"tpl1723365749212.png", record_pos=(0.324, -0.083), resolution=(3200, 1440)))
    touch(Template(r"tpl1723365768786.png", record_pos=(-0.398, -0.024), resolution=(3200, 1440)))
    touch(Template(r"tpl1723365781687.png", record_pos=(-0.276, -0.153), resolution=(3200, 1440)))
    yi_jian_ling_qu()
    touch(Template(r"tpl1723365975221.png", record_pos=(-0.189, -0.155), resolution=(3200, 1440)))

    touch(Template(r"tpl1723405587059.png", record_pos=(0.248, 0.181), resolution=(3200, 1440)))
    if exists(Template(r"tpl1723362476177.png", record_pos=(-0.002, 0.143), resolution=(3200, 1440))):
        touch(Template(r"tpl1723362476177.png", record_pos=(-0.002, 0.143), resolution=(3200, 1440)))
    yi_jian_ling_qu()
    touch(Template(r"tpl1723366166491.png", record_pos=(-0.102, -0.152), resolution=(3200, 1440)))
    yi_jian_ling_qu()
    touch(Template(r"tpl1723365009853.png", record_pos=(-0.402, -0.201), resolution=(3200, 1440)))
    sleep(1.0)
    touch(Template(r"tpl1723368156980.png", record_pos=(0.318, 0.198), resolution=(3200, 1440)))
    sleep(1.0)
    touch(Template(r"tpl1723368173825.png", record_pos=(-0.41, -0.102), resolution=(3200, 1440)))
    sleep(1.0)
    snapshot(msg="垃圾领完了，看看有啥好点的东西吗？")
    touch(Template(r"tpl1723365009853.png", record_pos=(-0.402, -0.201), resolution=(3200, 1440)))
    sleep(3)
    # 生成报告
    # simple_report(__file__, logpath=True, output=r"C:\Users\ntbgy\Desktop\log.html")


if __name__ == '__main__':
    # 设置日志级别
    logger_airtest = logging.getLogger("airtest")
    logger_ppocr = logging.getLogger("ppocr")
    logger_airtest.setLevel(logging.INFO)
    logger_ppocr.setLevel(logging.INFO)
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
    领垃圾(dir_path)
