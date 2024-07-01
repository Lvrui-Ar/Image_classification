import torchvision.models as models


def get_mobilenet_v3_large():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    return model
