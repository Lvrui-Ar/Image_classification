from models.RegNet.regnet_model import get_regnet_y_3_2gf
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'regnet_y_3_2gf'
    train_and_evaluate(model=get_regnet_y_3_2gf(), model_name=model_name)
