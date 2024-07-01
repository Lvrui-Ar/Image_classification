import torchvision.models as models


def get_resnet152():
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    return model
