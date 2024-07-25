import torchvision.models as models
from utils.parameters_save_util import create_directory


def get_models_original():
    densenet201 = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    efficientnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    regnet_y_3_2gf = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
    resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

    return {
        'densenet201':        densenet201,
        'efficientnet_v2_s':  efficientnet_v2_s,
        'mobilenet_v3_large': mobilenet_v3_large,
        'regnet_y_3_2gf':     regnet_y_3_2gf,
        'resnet152':          resnet152
    }


def print_models_architecture_o():
    models_dict = get_models_original()
    for key, value in models_dict.items():
        path = 'C:/Users/Lvrui/Desktop/project/classify/classify_test/models/Original_Arch/'
        create_directory(path)
        path += key + '.txt'

        with open(path, 'w', encoding='UTF-8') as f:
            f.write(str(value))
    return models_dict
