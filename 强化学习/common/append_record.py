import json


def append_record(file_path, data):
    """
    将数据以 JSON 格式追加到指定的文件中。

    参数:
        file_path (str): 要追加数据的文件路径。
        data (dict): 要追加的数据，必须是可 JSON 序列化的字典。

    返回:
        None

    异常:
        - 如果文件路径无效，抛出 ValueError。
        - 如果数据不是字典类型，抛出 TypeError。
    """
    # 以追加模式打开文件，并设置编码为 UTF-8
    with open(file_path, 'a', encoding='utf-8') as f:
        # 将数据转换为 JSON 字符串，并确保非 ASCII 字符能正确写入
        f.write(json.dumps(data, ensure_ascii=False))
        # 写入换行符，以便下次追加数据时从新行开始
        f.write('\n')
