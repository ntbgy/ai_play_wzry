# 导入 os 模块，用于与操作系统进行交互，例如路径操作
import os
# 导入 shutil 模块，用于高级文件操作，例如删除整个目录
import shutil

# 从 common.env 模块中导入 training_data_save_directory，这是训练数据的保存目录
from common.env import training_data_save_directory

# 打印训练数据保存目录，以便检查路径是否正确
print(training_data_save_directory)

# 获取训练数据保存目录下的所有文件夹名称
names = os.listdir(training_data_save_directory)

# 遍历所有文件夹
for name in names:
    # 构建当前文件夹的完整路径
    path = os.path.join(training_data_save_directory, name)

    # 获取当前文件夹中的所有文件，并过滤出以 .jpg 结尾的图片文件
    images = os.listdir(path)
    images = [item for item in images if '.jpg' in item]

    # 如果图片数量少于 800 张
    if len(images) <= 800:
        # 打印当前文件夹的名称，表示将删除该文件夹
        print(name)
        # 删除当前文件夹及其所有内容
        shutil.rmtree(path)
