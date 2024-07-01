import torchvision.models as models


def get_efficientnet_v2_s():
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    return model
