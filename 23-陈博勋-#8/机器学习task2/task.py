import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
'''
参数含义的解释
    batch_size是用于torch中的数据集加载器dataloader中，表示每次从数据集中抓取多少图片（本质是对tensor的操作）
    learning_rate即学习率，表示梯度下降过程中，参数调整的一个步长是多少，一个步长与梯度的乘积最后得到了在各个参数维度方向的调整值，
        通常learning_rate比较小，如0.001
    num_epochs是指训练的时候进行的轮数
    transform是从CIFAR_10数据集中获取数据的时候进行的预处理操作，这里的的代码进行的预处理操作是把图片数据的数组或者其他类型转化为tensor类型
        便于神经网络的计算
'''
batch_size = 64
learning_rate = 0.01
num_epochs = 100

transform = transforms.Compose([
    transforms.ToTensor()
])

# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.my_model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # TODO:这里补全你的前向传播
        img = self.my_model1(x)
        return img



# TODO:补全
model = Network().to(device)
# model = VGG().to(device)
criterion = nn.CrossEntropyLoss().to(device)   # 采用交叉熵损失函数用于分类任务
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 采用随机梯度下降进行参数更新和反向传播


def train():
    model.train()
    for epoch in range(num_epochs):
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

        accuracy = 100 * correct / total
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')


def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')


if __name__ == "__main__":
    train()
    test()
