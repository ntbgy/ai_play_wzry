import json
import os
import sqlite3

from common import get_now

# 定义训练数据样本的根目录
root_path = "E:/ai-play-wzry/训练数据样本/未用"
# 获取根目录下的所有文件名
names = os.listdir(root_path)

# 遍历所有文件名
for name in names:
    # 打开并读取 JSON 文件
    with open(f'{root_path}/{name}/_操作数据.json', 'r', encoding='utf-8') as f:
        data = f.read()
    # 去除数据中的空格并按行分割
    data = data.strip().split('\n')

    # 建立与数据库的连接
    conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
    # 创建游标对象
    cursor = conn.cursor()
    # 查询数据库中是否已存在该名称的数据
    s_sql = f"""
    select count(*) from training_full_data where name='{name}'
            """
    cursor.execute(s_sql)
    # 获取查询结果
    s_data = cursor.fetchall()
    # 如果数据条数与文件中的数据条数相同，则无需处理
    if int(s_data[0][0]) == len(data):
        print('无需处理', name)
        continue
    print('正在处理', name)
    # 遍历文件中的每一行数据
    for line in data:
        # 从 JSON 数据中获取图片名称
        image_name = json.loads(line)['图片号'] + '.jpg'
        # 查询数据库中是否已存在该图片的数据
        s_sql = f"""
select * from training_full_data where name='{name}' and image_name='{image_name}'
        """
        cursor.execute(s_sql)
        # 获取查询结果
        s_data = cursor.fetchall()
        # 如果数据已存在，则跳过
        if s_data:
            continue
        # 获取当前时间
        now = get_now()
        # 插入新数据到数据库
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
