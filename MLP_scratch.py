import torch
from torch import nn
from d2l import torch as d2l


batch_size = 256
# 从d2l库中加载Fashion-MNIST数据集，返回训练数据迭代器和测试数据迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
print(type(train_iter), type(test_iter))
# 输入特征数量（784，即28x28像素图像,
# 输出类别数量（10，即10种类型）,以及隐藏层单元数（256）
num_inputs, num_outputs, num_hiddens = 784, 10, 256
# W1, b1, W2, b2 分别是两个全连接层的权重矩阵和偏置向量,
# 使用nn.Parameter封装以便自动求导。
# params 列表包含了所有需要优化的参数。
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]
# 初始化参数，使其服从均值为0，标准差为0.01的正态分布
for param in params:
    nn.init.normal_(param, mean=0, std=0.01)


# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 多层感知机模型，包含输入层、隐藏层和输出层。其中：
# 输入层将图像拉平成一维向量。
# 隐藏层应用了ReLU激活函数。
# 输出层直接计算线性组合。
def net(X):
    X = X.reshape((-1, num_inputs))  # 256*724
    H = relu(torch.matmul(X, W1) + b1)  # 256*256
    # print(X.shape)
    # print(W1.shape)
    # print(H.shape)
    # print(b1.shape)
    return torch.matmul(H, W2) + b2


loss = nn.CrossEntropyLoss(reduction='none')

# num_epochs 和 lr 分别是训练轮次和学习率。
num_epochs, lr = 10, 0.1
# updater 使用torch.optim.SGD，即随机梯度下降优化器，用于更新网络参数。
updater = torch.optim.SGD(params, lr=lr)
# 训练模型。它接受模型、数据迭代器、损失函数、训练轮次和优化器作为参数,
# 执行训练过程并评估模型性能。
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()
