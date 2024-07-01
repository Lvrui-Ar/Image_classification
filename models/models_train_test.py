import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import get_cifar10_data
from utils.visualization_utils import log_tensorboard, save_results_to_file
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import time
import torchvision.transforms as transforms


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_and_evaluate(model, model_name, transform_func=get_transform(), epochs=50, batch_size=128,
                       learning_rate=0.001):

    # 使用GPU开始训练：
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 获取训练数据集：
    transform = transform_func    # 对输入图片的处理
    trainloader, testloader = get_cifar10_data(transform, batch_size=batch_size)

    model = model.to(device)
    model.train()

    # 损失的设定和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 可视化的展示：
    writer_base_path = '../../logs_epchos=50'
    writer_path = os.path.join(writer_base_path, model_name)
    writer = SummaryWriter(log_dir=writer_path)

    # 训练过程：
    start_time = time.time()  # 训练的起始时间：

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 训练过程中的损失的输出：
            if i % 256 == 255:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 256:.3f}')
                log_tensorboard(writer, 'Training Loss', running_loss / 256, epoch * len(trainloader) + i)
                running_loss = 0.0

        epoch_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Accuracy: {epoch_accuracy}%')
        log_tensorboard(writer, 'Training Accuracy', epoch_accuracy, epoch)

        # 内部评估过程：为了得出每epoch训练后的准确率
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        test_accuracy = 100 * correct / total
        print(f'Test Accuracy after Epoch {epoch + 1}: {test_accuracy}%')
        log_tensorboard(writer, 'Test Accuracy', test_accuracy, epoch)

    total_training_time = time.time() - start_time
    print(f'Total Training Time: {total_training_time} seconds')

    # 整体的评估过程：
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    final_accuracy = 100 * correct / total
    test_loss /= len(testloader)
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Final Test Accuracy: {final_accuracy}%')
    print(f'Final Test Loss: {test_loss}')

    # plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], title='Confusion Matrix')

    model_params = sum(p.numel() for p in model.parameters())
    print(f'Model Parameters: {model_params}')

    start_pred_time = time.time()
    _ = model(torch.randn(1, 3, 224, 224).to(device))
    pred_time = time.time() - start_pred_time
    print(f'Prediction Time for one image: {pred_time} seconds')

    metrics = {
        "Epoch": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,

        "Final Test Accuracy": final_accuracy,
        "Final Test Loss": test_loss,
        "Total Training Time": total_training_time,
        "Model Parameters": model_params,
        "Prediction Time for one image": pred_time,
    }

    model_name = model_name
    model_save_path = './' + model_name + '_results'

    save_results_to_file(model_save_path, model, metrics, cm, [str(i) for i in range(10)], [])
    writer.close()

    save_path = os.path.join(model_save_path, model_name + '.pth')
    save_path = './' + save_path
    torch.save(model.state_dict(), save_path)
