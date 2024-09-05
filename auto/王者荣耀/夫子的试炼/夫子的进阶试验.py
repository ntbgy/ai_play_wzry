# -*- encoding=utf8 -*-
__author__ = "ntbgy"

import ollama
from airtest.aircv import *
from airtest.core.api import *
from airtest.report.report import simple_report
from paddleocr import PaddleOCR

from common.airtestProjectsCommon import ocr_now_touch, get_now_img_txt


def get_local_qwen2_7b_answer(question):
    content = f"""
注意事项：
直接回答选择题答案，包括选项内容，不需要答案解析，不需要其他字符。
参考资料：
王者荣耀官方网站及游戏内的英雄介绍页面。
问题：
{question}
    """.strip()
    response = ollama.chat(
        model='qwen2:7b',
        messages=[
            {'role': 'user', 'content': content}
        ])
    answer = response['message']['content'].strip().replace(" ", "")
    print("#" * 50)
    print("question: ", content)
    print("=" * 50)
    print("answer: ", answer)
    print("#" * 50)
    return answer


def 进入答题界面():
    ocr_now_touch("战令")
    sleep(1)
    ocr_now_touch("任务")
    sleep(1)
    for _ in range(5):
        txt = get_now_img_txt()
        if "开始答题" in txt:
            ocr_now_touch("开始答题")
            sleep(1)
            return
        elif "继续答题" in txt:
            ocr_now_touch("继续答题")
            sleep(1)
            return
        else:
            swipe((1500, 1100), (1500, 500), duration=1.0)
            sleep(0.5)


def AI答题(dir_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 可以根据需要选择语言
    for _ in range(15):
        if exists(Template(r"谢谢老师.png", threshold=0.8, record_pos=(0.106, 0.081), resolution=(3200, 1440))):
            touch(Template(r"谢谢老师.png", threshold=0.8, record_pos=(0.106, 0.081), resolution=(3200, 1440)))
            break
        # 截屏当前画面
        screen = G.DEVICE.snapshot()
        # 局部截图
        screen = aircv.crop_image(screen, (1084, 258, 2660, 1124))
        # 保存局部截图到log文件夹中
        filename = try_log_screen(screen)['screen']
        # 使用PaddleOCR识别图片文字
        ocr_result = ocr.ocr(f"{dir_path}/log/{filename}", cls=True)
        question = ''
        for line in ocr_result:
            for word_info in line:
                # 获取识别结果的文字信息
                textinfo = word_info[1][0]
                if textinfo in [
                    '求助师父', '你还没有师傅哦'
                ]:
                    continue
                question += textinfo + '\n'

        answer = get_local_qwen2_7b_answer(question).replace("答案：", "")
        snapshot(filename='now.png')
        if not ocr_now_touch(answer):
            answers = answer.split(".")
            for answer in answers:
                res = ocr_now_touch(answer)
                if res:
                    break
            else:
                simple_report(__file__, logpath=True, output=f"{dir_path}/log/error_log.html")
                raise "这题做不下去了，点不到呢"
        sleep(7)
        touch((2660, 1124))


def 回到主页():
    touch(Template(r"tpl1723367630863.png", record_pos=(-0.41, -0.203), resolution=(3200, 1440)))
    sleep(1)


def 夫子的进阶试验(dir_path):
    进入答题界面()
    AI答题(dir_path)
    回到主页()


if __name__ == "__main__":
    auto_setup(
        __file__,
        logdir=True,
        devices=["Android:///", ],
    )
    夫子的进阶试验()
    # 生成报告
    simple_report(__file__, logpath=True, output=r".\log\log.html")
    # 获取当前文件绝对路径
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # 打开报告
    # os.startfile(f"{dir_path}\\log\\log.html")
