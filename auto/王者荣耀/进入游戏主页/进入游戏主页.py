from airtest.core.api import *


def 进入游戏主页():
    stop_app("com.tencent.tmgp.sgame")
    sleep(1.0)
    home()
    sleep(1.0)
    start_app("com.tencent.tmgp.sgame")
    sleep(20)
    if exists(Template(filename='确定.png')):
        touch(Template(filename='确定.png'))
        sleep(20)
    if exists(Template(r"tpl1724255354605.png", record_pos=(0.279, -0.163), resolution=(3200, 1440))):
        touch(Template(r"tpl1724255354605.png", record_pos=(0.279, -0.163), resolution=(3200, 1440)))
        sleep(1)
    wait(Template(r"tpl1723362143992.png", record_pos=(-0.005, 0.128), resolution=(3200, 1440)))
    touch(Template(r"tpl1723362143992.png", record_pos=(-0.005, 0.128), resolution=(3200, 1440)))
    if exists(Template(r"tpl1723363277037.png", record_pos=(-0.087, 0.1), resolution=(3200, 1440))):
        touch(Template(r"tpl1723363277037.png", record_pos=(-0.087, 0.1), resolution=(3200, 1440)))
    if exists(Template(r"tpl1723363315975.png", record_pos=(0.299, 0.049), resolution=(3200, 1440))):
        touch(Template(r"tpl1723363315975.png", record_pos=(0.299, 0.049), resolution=(3200, 1440)))
    for _ in range(15):
        if exists(Template(r"tpl1723362241684.png", record_pos=(0.345, -0.163), resolution=(3200, 1440))):
            touch(Template(r"tpl1723362241684.png", record_pos=(0.345, -0.163), resolution=(3200, 1440)))
            sleep(1.0)
        else:
            break
    sleep(1)


if __name__ == '__main__':
    auto_setup(
        __file__,
        logdir=True,
        devices=["android:///"]
    )
    进入游戏主页()
    # touch(Template(r"tpl1724255354605.png", record_pos=(0.279, -0.163), resolution=(3200, 1440)))
