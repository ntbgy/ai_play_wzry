import os

for root, dirs, files in os.walk('../myStudy'):
    print(dirs, type(dirs))
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
