import argparse
import datetime

import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist

model_path = './model_pth/vgg16_bn-6c64b313.pth'  # 预训练模型的数据储存文件

BATCH_SIZE = 500
LR = 0.0005
EPOCH = 50
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser(description='vgg')
parser.add_argument('--device_ids',type=str,default='0',help="Training Devices")
parser.add_argument('--local_rank',type=int,default=-1,help="DDP parameter,do not modify")
args = parser.parse_args()

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(

            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),


            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),


            nn.Linear(4096, num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
    """利用cfg，生成vgg网络每层结构的函数"""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:

            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)

    # 单机单进程多卡GPU
    gpus = [0,4,5,6,7]
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    return model


def getData():

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[1, 1, 1])])  #标准化公式为input[channel] =(input[channel] - mean[channel])/std[channel]
    trainset = tv.datasets.CIFAR10(root='./cifar_data', train=True, transform=transform, download=True)  # 获取CIFAR10的训练数据
    testset = tv.datasets.CIFAR10(root='./cifar_data', train=False, transform=transform, download=True)  # 获取CIFAR10的测试数据

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # 将数据集导入到pytorch.DataLoader中
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)  # 将测试集导入到pytorch.DataLoader中
    return train_loader, test_loader


def train():
    """创建网络，并开始训练"""
    start = datetime.datetime.now()

    trainset_loader, testset_loader = getData()  # 获取数据
    net = vgg16().cuda()  # 创建vgg16的网络对象
    net.train()
    print(net)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().cuda()  # 定义网络的损失函数句柄,CrossEntropyLoss内部已经包含了softmax过程，所以神经网络只需要直接输出需要输入softmax层的数据即可
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # 定义网络的优化器句柄，使用Adam优化器

    # Train the model
    for epoch in range(EPOCH):
        true_num = 0.
        sum_loss = 0.
        total = 0.
        accuracy = 0.
        for step, (inputs_cpu, labels_cpu) in enumerate(trainset_loader):
            inputs = inputs_cpu.cuda()
            labels = labels_cpu.cuda()
            output = net(inputs)
            loss = criterion(output, labels)  # 计算一次训练的损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传递
            optimizer.step()  # 用Adam优化器优化网络结构参数

            _, predicted = torch.max(output, 1)  # predicted 当前图像预测的类别
            sum_loss += loss.item()
            total += labels.size(0)
            accuracy += (predicted == labels).sum()
            # tensor数据(在GPU上计算的)如果需要进行常规计算，必须要加.cpu().numpy()转换为numpy类型，否则数据类型无法自动转换为float
            print("epoch %d | step %d: loss = %.4f, the accuracy now is %.3f %%." % (epoch, step, sum_loss/(step+1), 100.*accuracy.cpu().numpy()/total))
        # 检测当前网络在训练集上的效果，并显示当前网络的训练情况
        acc = test(net, testset_loader)
        print("")
        print("___________________________________________________")
        print("epoch %d : training accuracy = %.4f %%" % (epoch, 100 * acc))
        print("---------------------------------------------------")


    print('Finished Training')

    end = datetime.datetime.now()
    print(end - start)
    return net


def test(net, testdata):
    """检测当前网络的效果"""
    correct, total = .0, .0
    for inputs_cpu, labels_cpu in testdata:
        inputs = inputs_cpu.cuda()
        labels = labels_cpu.cuda()
        net.eval()  # 有些模块在training和test/evaluation的时候作用不一样，比如dropout等等。
                    # net.eval()就是将网络里这些模块转换为evaluation的模式
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        #print(predicted)
    # net.train()
    # tensor数据(在GPU上计算的)如果需要进行常规计算，必须要加.cpu().numpy()转换为numpy类型，否则数据类型无法自动转换为float
    return float(correct.cpu().numpy()) / total


if __name__ == '__main__':
    net = train()