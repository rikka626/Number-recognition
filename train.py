import torch
from torch import nn
from torch import optim

from model import Network

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

if __name__ == '__main__':
    # 图像的预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor()  # 转换为张量
    ])

    # 读入并构造数据集
    train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
    print("train_dataset length: ", len(train_dataset))

    # 小批量的数据读入
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    print("train_loader length: ", len(train_loader))

    model = Network()  # 模型本身，它就是我们设计的神经网络
    optimizer = optim.Adam(model.parameters())  # 优化模型中的参数
    criterion = nn.CrossEntropyLoss()  # 分类问题，使用交叉祯损失误应

    # 初始化存储训练过程的损失和准确率
    losses = []
    accuracy_list = []

    epochs = 3;
    # 进入模型的迭代循环
    for epoch in range(epochs):  # 外层循环，代表了整个训练数据集的遍历次数
        correct = 0
        total = 0

        # 内层循环使用train_loader，进行小批量的数据读取
        for batch_idx, (data, label) in enumerate(train_loader):
            # 前向传播
            output = model(data)  # 1.计算神经网络的前向传播结果
            loss = criterion(output, label)  # 2.计算output和标签label之间的损失loss

            # 反向传播
            loss.backward()  # 3.使用backward计算渐度
            optimizer.step()  # 4.使用optimizer.step更新参数
            optimizer.zero_grad()  # 5.将渐度清零

            # 统计损失
            losses.append(loss.item())

            # 计算训练集上的准确率
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
            if batch_idx % 100 == 0:
                print(f"Epoch {(epoch + 1)}/{epochs} "
                      f"| Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        # 计算并记录当前epoch的准确率
        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        print(f"Epoch {epoch + 1} Accuracy: {accuracy:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), 'mnist.pth')

    # 可视化损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

    # 可视化学习曲线（准确率）
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

    # 计算并显示混沌矩阵
    all_labels = []
    all_predictions = []
    all_images = []  # 存储所有图像

    # 遍历训练集来获取所有预测值和真实标签
    for data, label in train_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_images.extend(data.cpu().numpy())  # 将图像数据存储到列表中

    # 计算混沌矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)

    # 可视化混沌矩阵
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # 识别错误分类的图像
    errors = []
    for i in range(len(all_labels)):
        if all_labels[i] != all_predictions[i]:
            errors.append((all_images[i], all_labels[i], all_predictions[i]))

    # 可视化错误图像
    num_errors_to_show = 10  # 可选择显示的错误图像数量
    plt.figure(figsize=(15, 10))
    for i in range(min(num_errors_to_show, len(errors))):
        img, true_label, pred_label = errors[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img[0], cmap='gray')  # img[0] 是单通道图像
        plt.title(f'True: {true_label}, Pred: {pred_label}')
        plt.axis('off')

    plt.suptitle('Misclassified Images', fontsize=16)
    plt.show()
