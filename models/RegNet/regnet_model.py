import torchvision.models as models


def get_regnet_y_3_2gf():
    model = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
    return model
