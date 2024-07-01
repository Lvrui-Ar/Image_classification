from models.DenseNet.densenet_model import get_densenet201, get_densenet201_transform
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'densenet201'
    train_and_evaluate(transform_func=get_densenet201_transform(), model=get_densenet201(), model_name=model_name)


