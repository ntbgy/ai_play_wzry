import json
import os
import sqlite3

from common import get_now

root_path = "E:/ai-play-wzry/判断数据样本"

with open(f'{root_path}/判断新.json', 'r', encoding='utf-8') as f:
    data = f.read()
data = data.strip().split('\n')

# 建立与数据库的连接
conn = sqlite3.connect(r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db')
# 创建游标对象
cursor = conn.cursor()
for line in data:
    image_name = list(json.loads(line).keys())[0]
    s_sql = f"""select * from judge_state_data where image_name='{image_name}' """
    cursor.execute(s_sql)
    s_data = cursor.fetchall()
    if s_data:
        continue
    now = get_now()
    i_sql = f"""
INSERT INTO judge_state_data 
(root_path, image_name, judge_state_json, state_old, state_new, create_time, update_time)
VALUES('{root_path}', '{image_name}', '{line}', '{list(json.loads(line).values())[0]}', NULL, {now}, {now})
    """
    print(i_sql)
    # 执行 SQL 命令
    cursor.execute(i_sql)
    # 提交事务
    conn.commit()
# 关闭游标
cursor.close()
# 关闭数据库连接
conn.close()
