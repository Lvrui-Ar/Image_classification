from models.ConvNeXt.convnext_model import get_convnext_tiny, get_convnext_tiny_transform
from models.models_train_test import train_and_evaluate


if __name__ == "__main__":
    model_name = 'convnext_tiny'
    train_and_evaluate(transform_func=get_convnext_tiny_transform(), model=get_convnext_tiny(), model_name=model_name)
