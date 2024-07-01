from models.MobileNetV3.mobilenet_v3_model import get_mobilenet_v3_large
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'mobilenet_v3_large'
    train_and_evaluate(model=get_mobilenet_v3_large(), model_name=model_name)
