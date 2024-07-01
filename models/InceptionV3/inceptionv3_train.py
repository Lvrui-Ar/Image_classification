from models.InceptionV3.inceptionv3_model import get_inception_v3,get_inception_v3_transform
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'inception_v3'
    train_and_evaluate(model=get_inception_v3(), model_name=model_name, transform_func=get_inception_v3_transform())
