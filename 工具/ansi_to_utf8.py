import os

dir_path = r"E:\ai-play-wzry\训练数据样本\未用"
names = os.listdir(dir_path)
for name in names:
    file_path = f'{dir_path}\\{name}\\_操作数据.json'
    try:
        with open(file_path, 'r', encoding='ansi') as f:
            data = f.read()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        print(f'{file_path}, 已处理')
    except UnicodeDecodeError:
        print(f'{file_path}, 已有编码不是ansi，不处理')
