import os
from pathlib import Path

# 定义要创建的文件夹路径
folders_to_create = [
    'ConvNeXt',
    'DenseNet',
    'EfficientNetV2_S',
    'InceptionV3',
    'MNASNet1_3'
]

# 定义要创建的文件名
file_names = [
    '_model.py',
    '_test.py',
    '_train.py'
]

# 指定基础文件夹路径
base_path = '../models/'

# 遍历列表，创建文件夹
for folder in folders_to_create:
    # 构建文件夹完整路径，注意此处使用 folder 作为文件夹名称
    folders_path = os.path.join(base_path, folder)
    folder_path = os.path.join(folders_path, folder.lower() + '_results')

    # 创建文件夹
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # 输出创建的文件夹路径信息
    print(f"Created folder: {folder_path}")

    # 在文件夹下创建三个文件
    for file_name in file_names:
        file_path = os.path.join(folders_path, f'{folder.lower()}{file_name}')
        # 创建空文件
        Path(file_path).touch()

        # 输出创建的文件路径信息
        print(f"Created file: {file_path}")



