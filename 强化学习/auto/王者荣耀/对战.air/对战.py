import logging

from airtest.cli.parser import cli_setup

from common.airtestProjectsCommon import *


def 在线5V5(dir_path):
    txt = get_now_img_txt(dir_path)
    if '对战' in txt:
        pass
    elif ('继续' in txt
          or '返回大厅' in txt
          or '胜利' in txt
          or '失败' in txt
    ):
        结束对战回到首页(dir_path)
    else:
        raise ValueError('请先手工进入游戏首页，并手工进入开始匹配，但不用匹配游戏，再退出回到首页！')
    ocr_now_touch('对战', dir_path)
    sleep(1)
    ocr_now_touch('王者峡谷', dir_path)
    sleep(1)
    ocr_now_touch('人机', dir_path)
    sleep(1)
    ocr_now_touch('难度1', dir_path)
    sleep(1)
    ocr_now_touch('开始练习', dir_path)
    sleep(3)
    ocr_now_touch('开始匹配', dir_path)
    sleep(1)
    for i in range(10):
        txt = get_now_img_txt(dir_path)
        if '匹配成功' in txt:
            ocr_now_touch('确认', dir_path)
            sleep(3)
            break
        if i == 9:
            raise ValueError("进不去了", dir_path)
    touch((694, 666))
    sleep(1)
    ocr_now_touch('射手', dir_path)
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083341.png'))
    sleep(1)
    ocr_now_touch('确定', dir_path)
    for i in range(5):
        txt = get_now_img_txt(dir_path)
        if '确定' in txt:
            ocr_now_touch('确定', dir_path)
            sleep(2)
            break


def 结束对战回到首页(dir_path):
    sleep(5)
    ocr_now_touch('继续', dir_path)
    sleep(1)
    ocr_now_touch('继续', dir_path)
    sleep(1)
    ocr_now_touch('返回大厅', dir_path)
    sleep(1)
    ocr_now_touch('确定', dir_path)
    sleep(1)
    touch((0.5, 0.5))
    sleep(1)
    if '确定' in get_now_img_txt(dir_path):
        ocr_now_touch('确定', dir_path)
        sleep(1)
    if '对战' not in get_now_img_txt(dir_path):
        raise ValueError("结束对战回到首页 失败！")


def 离线1v1():
    # 需要先手工登录一个账号，退出了不行~
    txt = get_now_img_txt()
    if '开始游戏' not in txt:
        stop_app("com.tencent.tmgp.sgame")
        sleep(1.0)
        home()
        sleep(1.0)
        start_app("com.tencent.tmgp.sgame")
        sleep(20)
    touch(Template(filename='屏幕截图 2024-09-06 082752.png'))
    sleep(1)
    ocr_now_touch('开始练习')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083209.png'))
    sleep(1)
    ocr_now_touch('射手')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083341.png'))
    sleep(1)
    ocr_now_touch('挑选对手')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083209.png'))
    sleep(1)
    ocr_now_touch('射手')
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083529.png'))
    sleep(1)
    ocr_now_touch('开始对战')
    sleep(5)
    return '开始对战'


def 离线5V5(dir_path):
    # 需要先手工登录一个账号，退出了不行~
    ocr_now_touch('胜利', dir_path, sleep_time=3)
    ocr_now_touch('失败', dir_path, sleep_time=3)
    ocr_now_touch('继续', dir_path, sleep_time=3)
    ocr_now_touch('返回大厅', dir_path, sleep_time=3)
    txt = get_now_img_txt(dir_path)
    if '开始游戏' in txt:
        # touch(Template(filename='屏幕截图 2024-09-06 082752.png'))
        touch((2925, 399))
        sleep(2)
    elif '单机模式' in txt:
        print('当前已进入单机模式')
    elif '5V5模式' not in txt:
        # 直接重启吧~
        stop_app("com.tencent.tmgp.sgame")
        sleep(1.0)
        home()
        sleep(1.0)
        start_app("com.tencent.tmgp.sgame")
        sleep(30)
        if exists(Template(filename='tpl1724255354605.png')):
            touch(Template(filename='tpl1724255354605.png'))
        # touch(Template(filename='屏幕截图 2024-09-06 082752.png'))
        touch((2925, 399))
        sleep(2)
    ocr_now_touch('5V5模式', dir_path)
    sleep(1)
    ocr_now_touch('倔强青铜', dir_path)
    sleep(1)
    ocr_now_touch('开始练习', dir_path)
    sleep(1)
    # touch(Template(filename='屏幕截图 2024-09-06 083209.png'))
    touch((694, 666))
    sleep(1)
    if '发育路' in get_now_img_txt(dir_path):
        ocr_now_touch('发育路', dir_path)
    else:
        ocr_now_touch('射手', dir_path)
    sleep(1)
    touch(Template(filename='屏幕截图 2024-09-06 083341.png'))
    sleep(1)
    ocr_now_touch('确定', dir_path)
    sleep(1)
    ocr_now_touch('确定', dir_path)
    sleep(1)
    print('开始离线5V5')


def 已登录单人模式开始游戏(dir_path):
    ocr_now_touch('对战', dir_path, sleep_time=2)
    ocr_now_touch('王者峡谷', dir_path, sleep_time=2)
    ocr_now_touch('人机', dir_path, sleep_time=2)
    ocr_now_touch('单人模式', dir_path, sleep_time=1)
    ocr_now_touch('倔强青铜', dir_path, sleep_time=1)
    ocr_now_touch('开始练习', dir_path, sleep_time=3)
    touch(Template(filename='后裔.png'))
    sleep(2)
    ocr_now_touch('确定', dir_path, sleep_time=1)
    ocr_now_touch('确定', dir_path, sleep_time=10)


def 已登录单人模式返回大厅(dir_path):
    ocr_now_touch('继续', dir_path, sleep_time=3)
    ocr_now_touch('确定', dir_path, sleep_time=3)
    ocr_now_touch('继续', dir_path, sleep_time=3)
    ocr_now_touch('确定', dir_path, sleep_time=3)
    ocr_now_touch('继续', dir_path, sleep_time=3)
    ocr_now_touch('确定', dir_path, sleep_time=3)
    ocr_now_touch('返回大厅', dir_path, sleep_time=3)

def 在线发育路1v1(dir_path):
    ocr_now_touch('对战', dir_path, sleep_time=2)
    ocr_now_touch('1v1', dir_path, sleep_time=2)
    ocr_now_touch('人机', dir_path, sleep_time=2)
    ocr_now_touch('发育路', dir_path, sleep_time=3)
    # ocr_now_touch('后羿', dir_path, sleep_time=2, show_result=True)
    touch((290, 1200))
    sleep(2)
    ocr_now_touch('确定', dir_path, sleep_time=20, show_result=True)

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
    已登录单人模式返回大厅(dir_path)