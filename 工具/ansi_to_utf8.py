import os

# 定义要处理的目录路径
dir_path = r"E:\ai-play-wzry\训练数据样本\未用"

# 获取目录下的所有文件名
names = os.listdir(dir_path)

# 遍历文件名列表
for name in names:
    # 构建文件路径
    file_path = f'{dir_path}\\{name}\\_操作数据.json'

    # 尝试以 ANSI 编码读取文件
    try:
        with open(file_path, 'r', encoding='ansi') as f:
            data = f.read()
        # 将文件内容写入为 UTF-8 编码
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)
        # 验证文件是否已成功转换为 UTF-8 编码
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        print(f'{file_path}, 已处理')
    # 如果文件不是 ANSI 编码，捕获异常
    except UnicodeDecodeError:
        print(f'{file_path}, 已有编码不是ansi，不处理')
