import os
import time

from common.auto_game import get_img_txt, get_now_img
from common.my_logger import logger


# 检测游戏是否结束
def check_game_status(sp, flag_file_name):
    time.sleep(6 * 60)
    while True:
        if os.path.exists(flag_file_name):
            return
        txt = get_img_txt(get_now_img())
        keywords = ['返回大厅', '再来一局', '继续', '胜利', '失败', '请选择', '皮肤']
        if any(keyword in txt for keyword in keywords):
            logger.info("检测到游戏结束")
            sp.set_stop(True)
            break
        time.sleep(15)
