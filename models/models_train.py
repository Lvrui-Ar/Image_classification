from models.Training_Evaluation_process import train_and_evaluate
from utils.data.data_utils import get_cifar10_data, get_cifar100_data, get_weed_data
from models.Original_Arch.A_original_model_architecture import print_models_architecture_o
from models.Finetune_Arch.models_fine_tune import print_models_architecture_ft


def original_model(data_func, data_path):
    models_dict = print_models_architecture_o()

    for key1, value1 in models_dict.items():
        train_and_evaluate(value1, key1, data_func, data_path)


def finetune_model(data_func, data_name, data_path):
    tr, t, data_labels = data_func()
    model_dict = print_models_architecture_ft(data_labels, data_name)

    for key2, value2 in model_dict.items():
        train_and_evaluate(value2, key2, data_func, data_path)


if __name__ == '__main__':

    data_dict = dict(cifar10=get_cifar10_data, cifar100=get_cifar100_data, weeds=get_weed_data)
    # data_dict = dict(cifar100=get_cifar100_data, weeds=get_weed_data)
    for key, value in data_dict.items():
        O_path = 'Original_model architecture/'
        O_path = O_path+key+'/'
        FT_path = 'Fine-tune model architecture/'
        FT_path = FT_path+key+'/'

        original_model(value, O_path)
        finetune_model(value, key, FT_path)
