from models.ResNet152.resnet152_model import get_resnet152
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'resnet152'
    train_and_evaluate(model=get_resnet152(), model_name=model_name)
