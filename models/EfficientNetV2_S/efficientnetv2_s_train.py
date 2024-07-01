from models.EfficientNetV2_S.efficientnetv2_s_model import get_efficientnet_v2_s
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'efficientnet_v2_s'
    train_and_evaluate(model=get_efficientnet_v2_s(), model_name=model_name)


