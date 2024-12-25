# 导入 os 模块，用于与操作系统进行交互，例如路径操作
import os

# 使用 os.walk() 函数遍历指定目录 '../myStudy' 下的所有文件和子目录
for root, dirs, files in os.walk('../myStudy'):
    # 打印当前目录下的所有子目录名称，以及它们的类型（列表）
    print(dirs, type(dirs))
    # 遍历当前目录下的所有文件
    for file in files:
        # 使用 os.path.join() 函数拼接文件的完整路径
        file_path = os.path.join(root, file)
        # 打印文件的完整路径
        print(file_path)
