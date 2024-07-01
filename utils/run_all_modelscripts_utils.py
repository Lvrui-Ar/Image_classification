import os
import subprocess

# 定义要处理的文件夹和脚本
folders = ['../models/AlexNet/',
           '../models/ConvNeXt/',
           '../models/DenseNet/',
           '../models/EfficientNetV2_S/',
           '../models/InceptionV3/',
           '../models/MNASNet1_3/',
           '../models/MobileNetV3/',
           '../models/RegNet/',
           '../models/ResNet152/'
           ]

scripts = ['alexnet_train.py',
           'convnext_train.py',
           'densenet_train.py',
           'efficientnetv2_s_train.py',
           'inceptionv3_train.py',
           'mnasnet1_3_train.py',
           'mobilenetv3_train.py',
           'regnet_train.py',
           'resnet152_train.py'
           ]

# 遍历每个文件夹和脚本，依次执行
for folder, script in zip(folders, scripts):
    # 切换到目标文件夹
    os.chdir(folder)

    # 使用 subprocess.run() 来执行 Python 脚本
    result = subprocess.run(['python', script], capture_output=True, text=True)

    # 输出执行结果
    print('-' * 50)
    print(f"Executed {script} in {folder}")
    print("Output:")
    print(result.stdout)

    print("Errors:")
    print(result.stderr)
    print('-' * 50)

    # 返回上级目录
    os.chdir('..')
