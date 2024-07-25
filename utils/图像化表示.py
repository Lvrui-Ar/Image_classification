import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 创建数据框
data = {
    "Model Name": ["densenet201", "efficientnet_v2_s", "mobilenet_v3_large", "regnet_y_3_2gf", "resnet152"],
    "Training epochs": [21, 36, 31, 42, 24],
    "Batch size": [64, 64, 64, 64, 64],
    "Learning rate": [0.01, 0.01, 0.01, 0.01, 0.01],
    "Model parameters": ["20.01 M", "21.46 M", "5.48 M", "19.44 M", "60.19 M"],
    "Total training time (min)": [76.239, 84.286, 26.412, 143.563, 103.491],
    "Prediction time for one image (ms)": [110.44, 63.0, 41.0, 40.0, 50.0],
    "Final Training Accuracy": [99.87, 99.7, 99.83, 100, 100],
    "Final Training Loss": [0.008, 0.013, 0.009, 0.001, 0.001],
    "Final Validation Accuracy": [88.09, 88.9, 88.61, 89.52, 87.29],
    "Final Validation Loss": [0.567, 0.542, 0.632, 0.522, 0.674],
    "F1 Score (%)": [88.07, 88.88, 88.61, 89.49, 87.27],
    "Precision (%)": [88.12, 88.88, 88.62, 89.49, 87.28],
    "Recall (%)": [88.09, 88.90, 88.61, 89.52, 87.29]
}

df = pd.DataFrame(data)

# 设置图表风格
sns.set(style="whitegrid")

# 1. 训练过程和模型性能
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
sns.barplot(x="Model Name", y="Training epochs", data=df, ax=axes[0])
axes[0].set_title("Training epochs")

sns.barplot(x="Model Name", y="Total training time (min)", data=df, ax=axes[1])
axes[1].set_title("Total training time (min)")
plt.tight_layout()
plt.show()

# 2. 模型准确性和损失
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
sns.barplot(x="Model Name", y="Final Training Accuracy", data=df, ax=axes[0])
axes[0].set_title("Final Training Accuracy")

sns.barplot(x="Model Name", y="Final Validation Accuracy", data=df, ax=axes[1])
axes[1].set_title("Final Validation Accuracy")
plt.tight_layout()
plt.show()

# 3. 评估指标
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
sns.barplot(x="Model Name", y="F1 Score (%)", data=df, ax=axes[0])
axes[0].set_title("F1 Score (%)")

sns.barplot(x="Model Name", y="Precision (%)", data=df, ax=axes[1])
axes[1].set_title("Precision (%)")

sns.barplot(x="Model Name", y="Recall (%)", data=df, ax=axes[2])
axes[2].set_title("Recall (%)")
plt.tight_layout()
plt.show()
