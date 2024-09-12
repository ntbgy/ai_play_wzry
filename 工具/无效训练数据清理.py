import os
import shutil

from common.env import training_data_save_directory

print(training_data_save_directory)
names = os.listdir(training_data_save_directory)
for name in names:
    path = os.path.join(training_data_save_directory, name)
    images = os.listdir(path)
    images = [item for item in images if '.jpg' in item]
    if len(images) <= 800:
        print(name)
        shutil.rmtree(path)