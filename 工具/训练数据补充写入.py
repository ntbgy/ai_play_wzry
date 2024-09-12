import json
import os
import sqlite3
import time


def get_now():
    current_time = time.time()
    millisecond_part = str(int((current_time % 1) * 100)).zfill(2)
    time_struct = time.localtime(current_time)
    formatted_time = time.strftime('%Y%m%d%H%M%S', time_struct)
    return f"{formatted_time}{millisecond_part}"


root_path = "E:/ai-play-wzry/训练数据样本/未用"
names = os.listdir(root_path)
for name in names:
    with open(f'{root_path}/{name}/_操作数据.json', 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.strip().split('\n')
    # 建立与数据库的连接
    conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
    # 创建游标对象
    cursor = conn.cursor()
    print('正在处理', name)
    for line in data:
        image_name = json.loads(line)['图片号'] + '.jpg'
        s_sql = f"""
select * from training_full_data where name='{name}' and image_name='{image_name}'
        """
        cursor.execute(s_sql)
        data = cursor.fetchall()
        if data:
            continue
        now = get_now()
        i_sql = f"""
    INSERT INTO training_full_data 
    (name, root_path, action_touch, image_name, state, score, create_time, update_time) 
    VALUES('{name}', '{root_path}', '{line}', '{image_name}', NULL, NULL, {now}, {now});
        """
        # 执行 SQL 命令
        cursor.execute(i_sql)
        # 提交事务
        conn.commit()
    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    conn.close()
