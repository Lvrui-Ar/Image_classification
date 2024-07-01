import torchvision.models as models


def get_mnasnet1_3():
    model = models.mnasnet1_3(weights=models.MNASNet1_3_Weights.IMAGENET1K_V1)
    return model
