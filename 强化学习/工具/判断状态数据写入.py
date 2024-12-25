import json
import os
import sqlite3

from common import get_now

# 定义根目录路径
root_path = "E:/ai-play-wzry/判断数据样本"

# 打开并读取 JSON 文件
with open(f'{root_path}/判断新.json', 'r', encoding='utf-8') as f:
    data = f.read()
# 去除数据中的空格并按行分割
data = data.strip().split('\n')

# 建立与数据库的连接
conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
# 创建游标对象
cursor = conn.cursor()

# 遍历数据中的每一行
for line in data:
    # 从 JSON 字符串中获取 image_name
    image_name = list(json.loads(line).keys())[0]
    # 构造查询语句，检查数据库中是否已存在相同 image_name 的记录
    s_sql = f"""select * from judge_state_data where image_name='{image_name}' """
    # 执行查询语句
    cursor.execute(s_sql)
    # 获取查询结果
    s_data = cursor.fetchall()
    # 如果存在相同记录，则跳过当前循环
    if s_data:
        continue
    # 获取当前时间
    now = get_now()
    # 构造插入语句
    i_sql = f"""
INSERT INTO judge_state_data 
(root_path, image_name, judge_state_json, state_old, state_new, create_time, update_time)
VALUES('{root_path}', '{image_name}', '{line}', '{list(json.loads(line).values())[0]}', NULL, {now}, {now})
    """
    # 打印插入语句，方便检查
    print(i_sql)
    # 执行插入语句
    cursor.execute(i_sql)
    # 提交事务
    conn.commit()

# 关闭游标
cursor.close()
# 关闭数据库连接
conn.close()
