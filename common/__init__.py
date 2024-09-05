import os
from pathlib import Path


def get_dirs_path(dir_path):
    paths = list()
    for root, directory, files in os.walk(dir_path):
        paths.append(Path(root))
    return paths


def get_files_path(dir_path):
    paths = list()
    for root, directory, files in os.walk(dir_path):
        for file in files:
            paths.append(Path(root) / Path(file))
    return paths