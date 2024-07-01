import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def log_tensorboard(writer, tag, value, step):
    writer.add_scalar(tag, value, step)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()


def save_results_to_file(filepath, model, metrics, confusion_matrix, class_names, images):

    os.makedirs(filepath,exist_ok=True)

    filename = filepath + '/model_evaluation.txt'
    with open(filename, 'w') as f:
        f.write("Model Architecture:\n")
        f.write(str(model))
        f.write("\n\nMetrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=plt.cm.Blues,
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png')

    for i, img in enumerate(images):
        plt.imsave(f'{filepath}/image_{i}.png', img)
