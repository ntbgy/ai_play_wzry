import sqlite3

from common.my_logger import logger


def get_table_data(
        db_name=r'C:\Users\ntbgy\PycharmProjects\ai-play-wzry\data\AiPlayWzryDb.db',
        sql=None,
        table_name='training_data',
        condition=None
):
    # 连接到数据库
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 获取表头信息
    cursor.execute(f"PRAGMA table_info({table_name})")
    headers = [column[1] for column in cursor.fetchall()]

    # 根据条件查询数据
    if sql is None:
        if condition:
            sql = f"SELECT * FROM {table_name} WHERE {condition}"
        else:
            sql = f"SELECT * FROM {table_name}"
    logger.debug(sql)

    cursor.execute(sql)
    data = cursor.fetchall()

    # 关闭数据库连接
    conn.close()
    # 打印
    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = headers
    [table.add_row(item) for item in data]
    print(table)
    # 返回结果
    if data:
        return data
    else:
        return None


if __name__ == '__main__':
    res = get_table_data(condition="name = '1726126615'")
    print(res)
