import torch
import torch.nn as nn # type: ignore
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import islice


"""
ResNet-18的残差块
"""
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        """
        :param input_channels: 输入通道数
        :param num_channels: 输出通道数
        :param use_1x1conv: 是否使用1x1卷积来匹配输入输出通道数
        :param strides: 卷积步长,用于控制下采样的比例.
        """
        super().__init__()  # 父类的初始化方法
        # 定义残差块, 包含两个卷积层和一个BN层
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    # 前向传播方法
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """
    定义一个ResNet块，由多个残差块组成
    :param input_channels: 输入通道数
    :param num_channels: 输出通道数
    :param num_residuals: 残差块中残差单元的个数
    :param first_block: 是否是第一个残差块
    :return: 一个残差块
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 4个残差块
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# AdaptiveAvgPool2d自适应平均池化层，将输出的特征图大小调整为1x1，将特征图压缩成一个向量
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整到224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

lr, num_epochs = 0.01, 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型移动到设备上
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

def train_epoch(net, train_loader, loss, optimizer, device):
    """
    :param net:
    :param train_loader:
    :param loss:
    :param optimizer:
    :param device:
    :return:
    """
    net.train()
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")  # 添加进度条
    for X, y in progress_bar:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
        progress_bar.set_postfix(loss=train_l_sum / n, accuracy=train_acc_sum / n)  # 显示损失和准确率
    return train_l_sum / n, train_acc_sum / n

def test(net, test_loader, device):
    net.eval()
    test_acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            test_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]

    return test_acc_sum / n

# 初始化列表来记录每一轮的损失和准确率
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(net, train_loader, loss, optimizer, device)
    test_acc = test(net, test_loader, device)
    # 将当前epoch的损失和准确率添加到列表中
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    print(f'epoch {epoch + 1}, loss {train_loss:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}')

# 绘制训练损失和准确率
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

