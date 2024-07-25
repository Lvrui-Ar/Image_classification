import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.data.data_weeds import WeedDataset
from torch.utils.data import Dataset, DataLoader

transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def get_cifar10_data(transform=transform_base, batch_size=32):
    trainset = datasets.CIFAR10(root='C:/Users/Lvrui/Desktop/project/classify/classify_test/data', train=True,
                                download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='C:/Users/Lvrui/Desktop/project/classify/classify_test/data', train=False,
                               download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    cifar10_labels = trainset.classes

    return trainloader, testloader, cifar10_labels


def get_cifar100_data(transform=transform_base, batch_size=32):
    trainset = datasets.CIFAR100(root='C:/Users/Lvrui/Desktop/project/classify/classify_test/data', train=True,
                                 download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root='C:/Users/Lvrui/Desktop/project/classify/classify_test/data', train=False,
                                download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    cifar100_labels = trainset.classes

    return trainloader, testloader, cifar100_labels


def get_weed_data(transform=transform_base, batch_size=32):
    # 基础路径
    base_path = "C:/Users/Lvrui/Desktop/project/dataset/weeds_balanced"

    # 路径定义
    train_file_path = os.path.join(base_path, "balanced_train.txt")
    valid_file_path = os.path.join(base_path, "balanced_valid.txt")

    # 实例化数据集
    train_dataset = WeedDataset(file_path=train_file_path, transform=transform)
    valid_dataset = WeedDataset(file_path=valid_file_path, transform=transform)

    # 创建DataLoader
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    weed_labels = train_dataset.classes

    return trainloader, testloader, weed_labels


# get_cifar10_data()
# trainloader, testloader, cifar10_labels = get_cifar100_data()
# get_oxfordiiipet_data()
# get_weed_data()
