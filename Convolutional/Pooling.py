import torch
from torch import nn


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2), mode='max'))
print(pool2d(X, (2, 2), mode='avg'))


X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

# 深度学习框架中的步幅与汇聚窗口的大小相同
print("深度学习框架中的步幅与汇聚窗口的大小相同")
pool2d = nn.MaxPool2d(3)
print(pool2d(X))
# 填充和步幅可以手动设定
# kerneal_size: 是指窗口的大小，stride: 是指窗口在每一维的步幅，padding: 是指窗口在每一维的填充
print("填充和步幅可以手动设定")
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
# 设定一个任意大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度
# kernel_size: (height, width), stride: (height, width), padding: (height, width)

print("设定一个任意大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度")
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))
print("在处理多通道输入数据时，[汇聚层在每个输入通道上单独运算]")
X = torch.cat((X, X + 1), 1)
print(X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))