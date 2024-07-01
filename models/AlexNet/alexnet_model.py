import torchvision.models as models
import torchvision.transforms as transforms


def get_alexnet():
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return model


def get_alexnet_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
