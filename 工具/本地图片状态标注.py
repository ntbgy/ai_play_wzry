import json
import os
import random
import shutil
import threading
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pynput.keyboard import Key, Listener

from common import get_files_path, get_key_name, cv2ImgAddText, get_now, get_state_score, Transformer
from common.Batch import create_masks
from common.env import 判断状态模型地址, 状态词典B, 状态词典
from common.my_logger import logger
from common.resnet_utils import myResnet

态 = '暂停'

import sqlite3


# 监听按压
def on_press(key):
    global 态
    pass


# 监听释放
def on_release(key):
    global 态
    key_name = get_key_name(key)
    if key_name == 'Key.up':
        态 = '弃'
    elif key_name == 'Key.left':
        态 = '普通'
    elif key_name == 'Key.down':
        态 = '过'
    elif key_name == 'Key.right':
        态 = '死亡'
    elif key_name == 'a':
        态 = '击杀敌方英雄'
    elif key_name == 's':
        # 别太严格了，没补刀也算吧，毕竟没补刀也有经济，我方小兵或队友推塔也算吧
        态 = '击杀小兵或野怪或推掉塔'
    elif key_name == 'd':
        态 = '被击杀'
    elif key_name == 'w':
        态 = '被击塔攻击'
    if key == Key.esc:
        # 停止监听
        return False


# 开始监听
def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def 本地图片状态标注(file_root_dir, flag=True):
    global 态
    th = threading.Thread(target=start_listen, )
    th.daemon = True
    th.start()
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
    resnet101 = myResnet(mod)
    model_判断状态 = Transformer(6, 768, 2, 12, 0.0, 6 * 6 * 2048)
    model_判断状态.load_state_dict(torch.load(判断状态模型地址))
    model_判断状态.cuda(device)

    paths = get_files_path(file_root_dir)
    paths = [item for item in paths if '.jpg' in os.path.basename(item)]
    print(len(paths))
    if flag is True:
        paths = random.sample(paths, 10)
    得分求和 = 0
    for index, image_path in enumerate(paths):
        image_name = os.path.basename(image_path)
        image_dir = os.path.basename(os.path.dirname(image_path))
        image_new_name = image_dir + '_' + image_name.replace('.jpg', '')
        image_new_path = f'E:\\ai-play-wzry\\判断数据样本\\{image_new_name}.jpg'
        if os.path.isfile(image_new_path):
            continue
        image = Image.open(image_path)
        图片数组 = np.asarray(image)
        截屏 = torch.from_numpy(图片数组).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
        _, out = resnet101(截屏)
        out = torch.reshape(out, (1, 6 * 6 * 2048))
        操作序列A = np.ones((1, 1))
        操作张量A = torch.from_numpy(操作序列A.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(操作张量A.unsqueeze(0), 操作张量A.unsqueeze(0), device)
        outA = out.detach()
        实际输出, _ = model_判断状态(outA.unsqueeze(0), 操作张量A.unsqueeze(0), trg_mask)
        _, 抽样 = torch.topk(实际输出, k=1, dim=-1)
        抽样np = 抽样.cpu().numpy()
        状态列表 = []
        for K in 状态词典B:
            状态列表.append(K)
        状况 = 状态列表[抽样np[0, 0, 0, 0]]
        得分 = 状态词典[状况]
        得分求和 += 得分
        if flag is True:
            print(状况, 得分)
            if 状况 in ['普通', '死亡', '被击杀']:
                continue
            logger.info(状况)
            截图 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            截图 = cv2ImgAddText(截图, 状况, 0, 0, (000, 222, 111), 25)
            cv2.imshow('image', 截图)
            cv2.waitKey()

            while 态 == '暂停':
                time.sleep(0.02)
            新输出 = {}
            if 态 == '过':
                校准输出 = '过'
                continue
            elif 态 == '普通':
                校准输出 = '普通'
            elif 态 == '死亡':
                校准输出 = '死亡'
            elif 态 == '被击杀':
                校准输出 = '被击杀'
            elif 态 == '击杀小兵或野怪或推掉塔':
                校准输出 = '击杀小兵或野怪或推掉塔'
            elif 态 == '击杀敌方英雄':
                校准输出 = '击杀敌方英雄'
            elif 态 == '被击塔攻击':
                校准输出 = '被击塔攻击'
            elif 态 == '弃':
                态 = '暂停'
                continue
            else:
                logger.warning(f'{image_path}, {状况}')
                continue
            新输出 = {image_new_name: 校准输出}
            data_path = r'E:\ai-play-wzry\判断数据样本\判断新.json'
            if os.path.isfile(data_path):
                with open(data_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(新输出, ensure_ascii=False))
                    f.write('\n')
            else:
                with open(data_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(新输出, ensure_ascii=False))
                    f.write('\n')
            print(image_path, image_new_path)
            shutil.copy(image_path, image_new_path)
            print(状况, 校准输出)

    print(round(得分求和, 2))
    return round(得分求和, 2)
    # 态 = '暂停'
        # image = my_cv_imread(path)
        # # 检查图像是否成功读取
        # if image is not None:
        #     # 显示图像
        #     cv2.imshow('Image', image)
        #     # 等待按键按下，0 表示无限等待
        #     cv2.waitKey(0)
        #     # 关闭所有窗口
        #     cv2.destroyAllWindows()
        # else:
        #     print("无法读取图像")


def select_data(cursor, name):
    # 执行查询操作
    cursor.execute(f"select * from training_data where name = '{name}'")
    # 获取所有结果
    results = cursor.fetchall()
    if not results:
        return None
    for row in results:
        print(row)
    return results


def calculate_training_data_score():
    # 建立与数据库的连接
    conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
    # 创建游标对象
    cursor = conn.cursor()
    root_path = r"E:\ai-play-wzry\训练数据样本\未用"
    names = os.listdir(root_path)
    start_time = time.time()
    for name in names:
        if select_data(cursor, name) is not None:
            continue
        file_root_dir = f"{root_path}\\{name}"
        score = 本地图片状态标注(file_root_dir, False)
        sql = f"""INSERT INTO training_data (name,root_path,score,tag)
        VALUES ('{name}', '{root_path}', {score}, '未知')
        """
        # 执行 SQL 命令
        cursor.execute(sql)
        # 提交事务
        conn.commit()
        # 查询结果
        select_data(cursor, name)
        end_time = time.time()
        time_diff = end_time - start_time
        print("%.2f 秒" % time_diff)
    end_time = time.time()
    time_diff = end_time - start_time
    print("共 %.2f 秒" % time_diff)
    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    conn.close()


def status_annotation_from_training_full_data():
    while True:
        # 建立与数据库的连接
        conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
        # 创建游标对象
        cursor = conn.cursor()
        s_sql = """
    SELECT id, root_path, name, image_name
    FROM training_full_data WHERE state is NULL
    ORDER BY id DESC LIMIT 100
        """
        cursor.execute(s_sql)
        data = cursor.fetchall()
        if not data:
            break
        for index, line in enumerate(data):
            print(line)
            image_path = f"{line[1]}/{line[2]}/{line[3]}"
            if os.path.exists(image_path):
                state, score = get_state_score(image_path)
                u_sql = f"""UPDATE training_full_data SET state='{state}', score={score}, update_time='{get_now()}'
                WHERE id={line[0]}"""
            else:
                u_sql = f"""UPDATE training_full_data SET exist='False', update_time='{get_now()}'
                WHERE id={line[0]}"""
            # 执行 SQL 命令
            cursor.execute(u_sql)
            # 提交事务
            conn.commit()
        # 关闭游标
        cursor.close()
        # 关闭数据库连接
        conn.close()


def status_annotation_from_db():
    global 态
    th = threading.Thread(target=start_listen, )
    th.daemon = True
    th.start()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
    resnet101 = myResnet(mod)
    model_判断状态 = Transformer(6, 768, 2, 12, 0.0, 6 * 6 * 2048)
    model_判断状态.load_state_dict(torch.load(判断状态模型地址))
    model_判断状态.cuda(device)

    # 建立与数据库的连接
    conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
    # 创建游标对象
    cursor = conn.cursor()
    s_sql = """
        SELECT root_path, name, image_name
        FROM training_full_data WHERE state = '击杀敌方英雄' AND name = '1726153077'
        ORDER BY id DESC LIMIT 100
            """
    s_sql = """
    SELECT root_path, image_name 
    FROM judge_state_data
    WHERE state_old != state_new and state_new is not NULL
    """
    cursor.execute(s_sql)
    data = cursor.fetchall()
    for line in data:
        image_path = '/'.join(line)
        print(image_path)
        image_new_name = line[1] + '_' + line[2]
        image_new_path = f'E:\\ai-play-wzry\\判断数据样本\\{image_new_name}'
        if os.path.isfile(image_new_path):
            continue
        image = Image.open(image_path)
        图片数组 = np.asarray(image)
        截屏 = torch.from_numpy(图片数组).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
        _, out = resnet101(截屏)
        out = torch.reshape(out, (1, 6 * 6 * 2048))
        操作序列A = np.ones((1, 1))
        操作张量A = torch.from_numpy(操作序列A.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(操作张量A.unsqueeze(0), 操作张量A.unsqueeze(0), device)
        outA = out.detach()
        实际输出, _ = model_判断状态(outA.unsqueeze(0), 操作张量A.unsqueeze(0), trg_mask)
        _, 抽样 = torch.topk(实际输出, k=1, dim=-1)
        抽样np = 抽样.cpu().numpy()
        状态列表 = []
        for K in 状态词典B:
            状态列表.append(K)
        状况 = 状态列表[抽样np[0, 0, 0, 0]]
        logger.info(状况)
        截图 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        截图 = cv2ImgAddText(截图, 状况, 0, 0, (000, 222, 111), 25)
        cv2.imshow('image', 截图)
        cv2.waitKey()

        while 态 == '暂停':
            time.sleep(0.02)
        if 态 == '过':
            continue
        elif 态 == '普通':
            校准输出 = '普通'
        elif 态 == '死亡':
            校准输出 = '死亡'
        elif 态 == '被击杀':
            校准输出 = '被击杀'
        elif 态 == '击杀小兵或野怪或推掉塔':
            校准输出 = '击杀小兵或野怪或推掉塔'
        elif 态 == '击杀敌方英雄':
            校准输出 = '击杀敌方英雄'
        elif 态 == '被击塔攻击':
            校准输出 = '被击塔攻击'
        elif 态 == '弃':
            态 = '暂停'
            continue
        else:
            logger.warning(f'{image_path}, {状况}')
            continue
        新输出 = {image_new_name: 校准输出}
        data_path = r'E:\ai-play-wzry\判断数据样本\判断新.json'
        if os.path.isfile(data_path):
            with open(data_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(新输出, ensure_ascii=False))
                f.write('\n')
        else:
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(新输出, ensure_ascii=False))
                f.write('\n')
        print(image_path, image_new_path)
        shutil.copy(image_path, image_new_path)
        print(状况, 校准输出)


def status_annotation_from_judge_state_data():
    while True:
        # 建立与数据库的连接
        conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
        # 创建游标对象
        cursor = conn.cursor()
        s_sql = """
    SELECT id, root_path, image_name, state_old
    FROM judge_state_data
    ORDER BY id DESC LIMIT 100
        """
        cursor.execute(s_sql)
        data = cursor.fetchall()
        if not data:
            break
        for index, line in enumerate(data):
            if '.jpg' in line[2]:
                image_name = line[2]
            else:
                image_name = line[2] + '.jpg'
            image_path = f"{line[1]}/{image_name}"
            print(image_path)
            if os.path.exists(image_path):
                state, _ = get_state_score(image_path)
                u_sql = f"""UPDATE judge_state_data
                SET state_new='{state}', update_time='{get_now()}'
                WHERE id={line[0]}"""
            else:
                u_sql = f"""UPDATE judge_state_data 
                SET exist='False', update_time='{get_now()}'
                WHERE id={line[0]}"""
            # 执行 SQL 命令
            cursor.execute(u_sql)
            # 提交事务
            conn.commit()
        # 关闭游标
        cursor.close()
        # 关闭数据库连接
        conn.close()
if __name__ == '__main__':
    status_annotation_from_judge_state_data()
