import torchvision.models as models
import torchvision.transforms as transforms

def get_inception_v3():
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    return model


def get_inception_v3_transform():
    return transforms.Compose([
        transforms.Resize(350),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
