import os

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练相关参数（全局变量）(当main函数没有设置的时候)
batch_size = 64
learning_rate = 0.01
num_epochs = 100
max_accuracy = 0.0

# 模型类的定义
class VGG_CIFAR10(nn.Module):
    def __init__(self):
        super(VGG_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 模型搭建（使用timm）
# VIT模型(微调)
model_VIT_pre = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
model_VIT = VisionTransformer(img_size=224, patch_size=16, num_classes=10, embed_dim=128, depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True)
model_VIT_pre_patch16 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes = 10)
model_VIT_pre_patch32 = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes = 10)
# VGG(微调)
model_VGG_pre = timm.create_model('vgg16', pretrained=True, num_classes=10)
model_VGG = VGG_CIFAR10()


def train():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # 若模型输入尺寸224*224，进行resize，
        if if_resize:
            inputs = interpolate(inputs, size=(224, 224))
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
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
    # 可视化展示
    writer.add_scalar("train_loss", running_loss / len(trainloader), epoch)
    writer.add_scalar("train_accuracy", accuracy, epoch)


def test():
    model.eval()
    correct = 0
    total = 0
    global max_accuracy
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # 若模型输入尺寸224*224，进行resize，
            if if_resize:
                images = interpolate(images, size=(224, 224))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
    # 可视化展示
    writer.add_scalar("test_accuracy", accuracy, epoch)
    accuracy_array.append(accuracy)
    # 模型保存
    model_path = os.path.join(result_path,"model_parameter", 'epoch{}_accuracy{}.pth'.format(epoch, accuracy))
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        torch.save(model.state_dict(), max_model_path)
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    # 相关参数
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 5
    index = 1
    experiment_label = "{}".format(index)

    # （重要！图片尺寸调整）当模型输入尺寸为22*224时，在数据处理中调用函数，利用插值法进行修改
    if_resize = True

    # 模型、优化器、损失函数
    model_name = "VIT"
    parameter_label = "pretrained_patch16" + "_" + "epoch={}".format(num_epochs)
    model = model_VIT_pre_patch16
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 准备tensorboard, 注意模型和参数更改要更改前面的
    result_dir = "runs"
    result_path = os.path.join(result_dir, model_name, parameter_label, experiment_label)
    while os.path.exists(result_path):
        index += 1
        experiment_label = "{}".format(index)
        result_path = os.path.join(result_dir, model_name, parameter_label, experiment_label)
    print("result_path: tensorboard & model parameter = " + result_path)
    writer = SummaryWriter(log_dir=result_path)
    model_path = os.path.join(result_path, "model_parameter")
    max_model_path = os.path.join(result_path, "model_parameter", 'max_accuracy.pth')
    try:
        os.mkdir(model_path)
        print("成功创建model_path目录")
    except FileExistsError:
        print(f"目录 {model_path} 已存在")

    # 准备matplotlib,进行patch不同值的对比试验
    accuracy_array = []

    max_accuracy = 0.0
    for epoch in range(num_epochs):
        train()
        test()

    plt.plot(range(1, 6), accuracy_array, 'r--', label=parameter_label)

    # 对max_accuracy进行重命名，标注出最终的最大accuracy(还未测试，重命名是否会报错，因为之前没有使用重命名，在kaggle上训练的)
    max_model_path_with_accuracy_label = os.path.join(result_path, "model_parameter", 'max_accuracy={}.pth'.format(max_accuracy))
    os.rename(max_model_path, max_model_path_with_accuracy_label)
    writer.close()

    # 对比不同patch试验的时候加入的代码
    model_name = "VIT"
    parameter_label = "pretrained_patch32" + "_" + "epoch={}".format(num_epochs)
    model = model_VIT_pre_patch32
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 准备tensorboard, 注意模型和参数更改要更改前面的
    result_dir = "runs"
    result_path = os.path.join(result_dir, model_name, parameter_label, experiment_label)
    while os.path.exists(result_path):
        index += 1
        experiment_label = "{}".format(index)
        result_path = os.path.join(result_dir, model_name, parameter_label, experiment_label)
    print("result_path: tensorboard & model parameter = " + result_path)
    writer = SummaryWriter(log_dir=result_path)
    model_path = os.path.join(result_path, "model_parameter")
    max_model_path = os.path.join(result_path, "model_parameter", 'max_accuracy.pth')
    try:
        os.mkdir(model_path)
        print("成功创建model_path目录")
    except FileExistsError:
        print(f"目录 {model_path} 已存在")

    # 准备matplotlib,进行patch不同值的对比试验
    accuracy_array = []

    max_accuracy = 0.0
    for epoch in range(num_epochs):
        train()
        test()

    plt.plot(range(1, 6), accuracy_array, 'r--', label=parameter_label)
    # 对max_accuracy进行重命名，标注出最终的最大accuracy(还未测试，重命名是否会报错，因为之前没有使用重命名，在kaggle上训练的)
    max_model_path_with_accuracy_label = os.path.join(result_path, "model_parameter",
                                                      'max_accuracy={}.pth'.format(max_accuracy))
    os.rename(max_model_path, max_model_path_with_accuracy_label)
    writer.close()

    # 对比结果图保存
    plt.title('Comparison of patch16 and patch32')
    plt.xlabel("epoch")
    plt.ylabel("test_accuracy")
    comparison_path = os.path.join(result_dir, model_name, "comparison_in_patch.png")
    plt.savefig(comparison_path)

    # 转化result目录，用于把两个模型的结果都保存下来
    result_path = comparison_path = os.path.join(result_dir, model_name)








    '''
    遇到的问题，解决方法
    max_accuracy模型参数结果保存了多个，因为附带了accuracy的信息，导致每次名称不同，没有进行覆盖
    
    '''