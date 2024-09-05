import os
from pathlib import Path

from airtest.core.api import touch, auto_setup, snapshot
from airtest.report.report import simple_report
from paddleocr import PaddleOCR


# ocr = PaddleOCR()
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 可以根据需要选择语言
def ocr_touch(target_text, pic_path):
    # 使用PaddleOCR识别图片文字
    ocr_result = ocr.ocr(img=str(pic_path), cls=True)
    # 遍历识别结果，找到目标文字的坐标
    target_coords = None
    txt = ''
    for line in ocr_result:
        for word_info in line:
            # 获取识别结果的文字信息
            textinfo = word_info[1][0]
            txt += textinfo + '\n'
            if target_text in textinfo:
                # 获取文字的坐标（中心点）
                x1, y1 = word_info[0][0]
                x2, y2 = word_info[0][2]
                target_coords = ((x1 + x2) / 2, (y1 + y2) / 2)
                break
        if target_coords:
            break
    # 使用Airtest点击坐标
    if target_coords:
        touch(target_coords)
        return True
    else:
        print(f"未找到目标文字：{target_text}")
        print(f"识别文字：{txt}")
        return False


if __name__ == '__main__':
    auto_setup(
        __file__,
        logdir=True,
        devices=["android:///", ],
    )
    # pic_path = r"C:\Users\ntbgy\PycharmProjects\python3Project\airtestProjects\王者荣耀\夫子的试炼\log\1723541269213.jpg"
    pic_path = Path(os.getcwd()) / 'log/now.png'
    snapshot(filename='now.png')
    ocr_touch("A.一次五连抽", pic_path)
    # 生成报告
    simple_report(__file__, logpath=True, output=r"C:\Users\ntbgy\Desktop\log.html")
