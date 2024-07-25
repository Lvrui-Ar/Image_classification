import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 自定义数据集类
class WeedDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path  # 文件路径
        self.transform = transform  # 数据转换
        self.image_labels = self._read_file(file_path)  # 读取文件并获取图像和标签
        self.classes = ['Chineseapple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly acacia',
                        'Rubber vine', 'Siam weed', 'Snake weed', 'Negatives']

    def __len__(self):
        return len(self.image_labels)  # 返回数据集大小

    def __getitem__(self, idx):
        img_path, label = self.image_labels[idx]  # 获取图像路径和标签
        image = Image.open(img_path).convert("RGB")  # 打开图像并转换为RGB模式

        if self.transform:
            image = self.transform(image)  # 进行数据转换

        return image, label  # 返回图像和标签

    @staticmethod
    def _read_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()  # 读取文件中的所有行
        # 将相对路径转换为绝对路径，并获取图像路径和标签
        image_labels = []
        for line in lines:
            parts = line.strip().split()
            img_path = os.path.abspath(os.path.join(base_path, parts[0]))  # 获取绝对路径
            label = int(parts[1])
            image_labels.append((img_path, label))

        return image_labels


# 定义数据转换
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


# 读取标签映射
def read_label_list(label_list_path):
    with open(label_list_path, 'r') as f:
        lines = f.readlines()  # 读取文件中的所有行
    # 解析标签映射
    label_map = {int(line.split(')')[0][1:]): line.split(')')[1].strip() for line in lines}
    label_list = []
    for i in range(len(label_map)):
        label_list.append(label_map[i])
    return label_list


# 基础路径
base_path = "C:/Users/Lvrui/Desktop/project/dataset/weeds_balanced"

# 路径定义
train_file_path = os.path.join(base_path, "balanced_train.txt")
valid_file_path = os.path.join(base_path, "balanced_valid.txt")
label_list_path = os.path.join(base_path, "label_list.txt")
