from models.AlexNet.alexnet_model import get_alexnet, get_alexnet_transform
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'alexnet'
    train_and_evaluate(transform_func=get_alexnet_transform(), model=get_alexnet(), model_name=model_name,epochs=50)
