import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

from utils.parameters_save_util import parameters_save


# 数据增强操作
def get_transform():
    """
    定义数据预处理的变换，包括图像缩放、随机水平翻转、随机旋转、中心裁剪、转为张量、以及标准化。
    返回：
        transforms.Compose: 包含一系列图像预处理操作的组合
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch):
    """
    训练模型一个epoch，并记录训练过程中的损失和准确率。
    """
    model.train()
    running_loss = 0.0  # 用于累计损失值
    correct = 0  # 记录预测正确的样本数
    total = 0  # 用于记录样本总数
    epoch_loss = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播，计算损失，反向传播，优化参数
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累积损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        count = i % 200
        # 每50个batch打印一次损失并记录到TensorBoard
        if count == 199:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

        epoch_loss = round(running_loss/(count+1), 3)

    # 记录每个epoch的训练准确率、损失
    epoch_accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1}, Accuracy: {epoch_accuracy:.3f}%, Loss: {epoch_loss}')
    epoch_accuracy = round(epoch_accuracy, 2)

    return epoch_accuracy, epoch_loss


def evaluate_model(model, testloader, criterion, device, epoch):
    """
    在测试集上评估模型，并记录测试过程中的准确率、损失和混淆矩阵。
    """
    model.eval()
    correct, total, test_loss, all_labels, all_preds = (0, 0, 0.0, [], [])

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

    # 记录测试准确率、损失、F1
    test_accuracy = 100 * correct / total
    test_loss /= len(testloader)
    test_loss = round(test_loss, 3)

    # 打印
    print(f'Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.3f}%')

    return test_accuracy, test_loss, all_labels, all_preds


def calculate_model_metrics(model, device):
    """
    计算模型的参数量和单张图片的预测时间。
    返回：
        tuple: 模型的参数量、单张图片的预测时间
    """
    # 计算模型参数量
    model_params = sum(p.numel() for p in model.parameters())
    model_params = str(round(model_params/1e6, 2))+' M'

    # 计算单张图片的预测时间
    start_pred_time = time.time()
    _ = model(torch.randn(1, 3, 224, 224).to(device))
    pred_time = round((time.time() - start_pred_time)*1000, 2)

    return model_params, pred_time


def train_and_evaluate(model, model_name, data_func, save_dir="test", epochs=50, batch_size=64):
    """
    训练并评估模型，并保存训练过程中的日志和最终的模型。
    """

    # 设定参数：
    test_accuracy, test_loss, f1, train_epoch, all_labels, all_preds = (0, 0, 0, 0, [], [])
    epoch_accuracy_list, epoch_loss_list, test_accuracy_list, test_loss_list = ([], [], [], [])
    learning_rate = 0.01
    step_size = 10
    gamma = 0.1   # 新增参数
    transform_func = get_transform()

    # 1. 设置设备（GPU或CPU）并打印使用的设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 2. 获取训练和测试数据集，并应用数据预处理。
    transform = transform_func
    trainloader, testloader, labels = data_func(transform, batch_size=batch_size)

    # 3. 将模型转移到设备上（GPU或CPU）
    model = model.to(device)

    # 4. 定义损失函数（交叉熵损失）和优化器（NAdam优化器）。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # 添加学习率调度器

    # 6. 记录训练的起始时间。
    total_training_time = 0

    # best_val_loss = float('inf')  # 初始化最佳验证集损失为无穷大
    best_test_accuracy = 0.0
    early_stopping_patience = 10  # 早停法的耐心值
    early_stopping_counter = 0  # 早停法计数器

    #  7. 在每个epoch中：- 训练模型，并记录训练损失和准确率。
    #     - 在测试集上评估模型，并记录测试损失和准确率。 - 记录每个epoch的训练时间。
    for epoch in range(epochs):
        train_epoch = epoch
        epoch_start_time = time.time()
        # 训练模型一个epoch
        epoch_accuracy, epoch_loss = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)
        # 学习率调整
        scheduler.step()  # 每个epoch结束后调整学习率
        # 记录每个epoch的训练时间
        total_training_time += time.time() - epoch_start_time
        # 在测试集上评估模型
        test_accuracy, test_loss, all_labels, all_preds = evaluate_model(
            model, testloader, criterion, device, epoch)

        epoch_accuracy_list.append(epoch_accuracy)
        epoch_loss_list.append(epoch_loss)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy  # 更新最佳验证集损失,更新为最新准确率
            early_stopping_counter = 0  # 重置早停法计数器
        else:
            early_stopping_counter += 1  # 未改善则计数器加1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. No improvement in validation loss.")  # 触发早停法
            break

    # 8. 计算总的训练时间并打印。
    print(f'Total Training Time (minutes): {total_training_time/60:.3f} minutes')

    # 9. 计算模型的参数量和单张图片的预测时间并打印。
    model_params, pred_time = calculate_model_metrics(model, device)

    # 保存训练数据和模型
    metrics = {
        "Epoch": train_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "Total Training Time": str(round(total_training_time/60, 3))+' min',
        "Model Parameters": model_params,
        "Prediction Time for one image": str(pred_time)+'ms',

        "epoch_accuracy_list": epoch_accuracy_list,
        "epoch_loss_list": epoch_loss_list,
        "test_accuracy_list": test_accuracy_list,
        "test_loss_list": test_loss_list,

        "all_labels": all_labels,
        "all_preds": all_preds
    }

    model_save_dir = 'C:/Users/Lvrui/Desktop/project/classify/classify_test/doc/'
    model_save_path = os.path.join(model_save_dir, save_dir, model_name)
    parameters_save(model_save_path, metrics, model_name, labels)

