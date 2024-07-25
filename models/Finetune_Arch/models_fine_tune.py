import random
import torch.nn as nn
import torchvision.models as models

from utils.parameters_save_util import create_directory


def FineTuneDenseNet(classes_num):
    # 加载预训练模型
    densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

    # 冻结特征提取层的参数
    for param in densenet.parameters():
        param.requires_grad = False

    # 自定义classifier部分，可以在这里修改
    densenet.classifier = nn.Sequential(
        nn.Linear(in_features=1920, out_features=256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, classes_num)
    )
    return densenet


def FineTuneEfficientNet(classes_num):
    # 加载预训练模型
    efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # 冻结特征提取层的参数
    for param in efficientnet.parameters():
        param.requires_grad = False

    # 自定义classifier部分，可以在这里修改
    efficientnet.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, classes_num)
    )
    return efficientnet


def FineTuneMobileNet(classes_num):
    # 加载预训练模型
    mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    # 冻结特征提取层的参数
    for param in mobilenet.parameters():
        param.requires_grad = False

    # 自定义classifier部分，可以在这里修改
    mobilenet.classifier = nn.Sequential(
        nn.Linear(in_features=960, out_features=1280, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=256, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=256, out_features=classes_num, bias=True)
    )
    return mobilenet


def FineTuneRegNet(classes_num):
    # 加载预训练模型
    regnet = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)

    # 冻结特征提取层的参数
    for param in regnet.parameters():
        param.requires_grad = False

    # 自定义classifier部分，可以在这里修改
    regnet.fc = nn.Sequential(
        nn.Linear(in_features=1512, out_features=256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(256, classes_num)
    )
    return regnet


def FineTuneResNet(classes_num):
    # 加载预训练模型
    resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

    # 冻结特征提取层的参数
    for param in resnet.parameters():
        param.requires_grad = False

    # 自定义classifier部分，可以在这里修改
    resnet.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=1024, out_features=256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=256, out_features=classes_num, bias=True),
    )
    return resnet


def get_models_fine_tune(label):
    classes_num = len(label)

    densenet201_ft = FineTuneDenseNet(classes_num)
    efficientnet_v2_s_ft = FineTuneEfficientNet(classes_num)
    mobilenet_v3_large_ft = FineTuneMobileNet(classes_num)
    regnet_y_3_2gf_ft = FineTuneRegNet(classes_num)
    resnet152_ft = FineTuneResNet(classes_num)

    return {
        'densenet201_ft':        densenet201_ft,
        'efficientnet_v2_s_ft':  efficientnet_v2_s_ft,
        'mobilenet_v3_large_ft': mobilenet_v3_large_ft,
        'regnet_y_3_2gf_ft':     regnet_y_3_2gf_ft,
        'resnet152_ft':          resnet152_ft
    }


def print_models_architecture_ft(label, name):
    model_dict = get_models_fine_tune(label)

    for key, value in model_dict.items():
        path = ('C:/Users/Lvrui/Desktop/project/classify/classify_test/models/Finetune_Arch' +
                f"/{name}/")
        create_directory(path)
        path += key + '.txt'

        with open(path, 'w', encoding='UTF-8') as f:
            f.write(str(value))

    return model_dict


# labels = [random.randint(1, 100) for _ in range(10)]
# print_models_architecture_ft(labels)
