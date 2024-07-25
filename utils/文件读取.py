import os
import pandas as pd


def merge_excel_sheets(basedir, outputfile):
    # 初始化一个空的字典来存储每个模型的数据
    data_dict = {}

    # 遍历基目录中的每个文件夹
    for folder in os.listdir(basedir):
        folder_path = os.path.join(basedir, folder)

        # 检查路径是否为目录
        if os.path.isdir(folder_path):
            # 定义Excel文件的路径
            file_path = os.path.join(folder_path, "model_metrics.xlsx")

            # 检查Excel文件是否存在
            if os.path.exists(file_path):
                # 从第一个工作表读取数据
                df = pd.read_excel(file_path, sheet_name="Parameters", header=None)
                # 将第一列作为索引
                df.set_index(0, inplace=True)
                # 将数据存储到字典中，以文件夹名作为键
                data_dict[folder] = df[1]

    # 将字典转换为DataFrame
    all_data = pd.DataFrame(data_dict)

    # 将合并的数据保存到一个新的Excel文件中
    all_data.to_excel(outputfile)


# 定义基目录路径和输出文件名
base_dir = "C:/Users/Lvrui/Desktop/project/classify/classify_test/doc/"  # 替换为您的实际路径
dir_list1 = ['Original_model architecture/', 'Fine-tune model architecture/']
dir_list2 = ['cifar10', 'cifar100', 'weeds']


# 调用函数合并Excel工作表
for dir1 in dir_list1:
    for dir2 in dir_list2:
        dirpath = base_dir + dir1 + dir2
        output_file = dirpath + f"/{dir2}_metrics.xlsx"  # 替换为您的实际路径
        merge_excel_sheets(dirpath, output_file)
