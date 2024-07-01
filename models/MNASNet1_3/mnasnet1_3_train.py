from models.MNASNet1_3.mnasnet1_3_model import get_mnasnet1_3
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'mnasnet1_3'
    train_and_evaluate(model=get_mnasnet1_3(), model_name=model_name)
