import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def get_cifar10_data(transform=transform_base, batch_size=32):
    trainset = datasets.CIFAR10(root='C:/Users/Lvrui/Desktop/project/classifiy/classifiy_test/data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='C:/Users/Lvrui/Desktop/project/classifiy/classifiy_test/data', train=False,
                               download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def get_oxfordiiipet_data(transform=transform_base, batch_size=32):
    trainset = datasets.OxfordIIITPet(root='C:/Users/Lvrui/Desktop/project/classifiy/classifiy_test/data',
                                      split='trainval', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.OxfordIIITPet(root='C:/Users/Lvrui/Desktop/project/classifiy/classifiy_test/data', split='test',
                                     download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# get_cifar10_data()
get_oxfordiiipet_data()
