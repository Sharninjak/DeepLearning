import torch
from torch import nn
from d2l import torch as d2l

# 定义数据集
# 定义数据集的大小，特征数量和批量大小
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 真实的权重和偏置，用于生成合成数据
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 生成训练数据
train_data = d2l.synthetic_data(true_w, true_b, n_train)
# 将训练数据加载到迭代器中，用于训练
train_iter = d2l.load_array(train_data, batch_size)
# 生成测试数据
test_data = d2l.synthetic_data(true_w, true_b, n_test)
# 将测试数据加载到迭代器中，用于评估模型
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


def init_params():
    """
    初始化模型参数，包括权重和偏置。
    权重初始化为标准正态分布，偏置初始化为0。
    :return: 权重w和偏差b
    """
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l1_penalty(w):
    """
    计算权重的L1范数作为正则化项，以避免稀疏权重
    :param w: 权重
    :return: L1范数惩罚项
    """
    return torch.sum(torch.abs(w))


# L2
def l2_penalty(w):
    """
    计算权重的L2范数作为正则化项，以避免过拟合
    :param w: 权重
    :return: L2范数惩罚项
    """
    return torch.sum(w.pow(2)) / 2  # w^2/2


def train_L1(lambd):

    # 初始化模型参数,定义模型和损失函数,设置训练轮数和学习率
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.004
    # 创建动画器，用于绘制损失曲线
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l1_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        # 每隔5个epoch，绘制当前模型在训练集和测试集上的损失
        if (epoch + 1) % 5 == 0:
            # 在动画器中绘制训练损失和测试损失
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
        # print('w的L2范数是：', torch.norm(w).item())
    print('w的L1范数是：', torch.norm(w).item())


def train_L2(lambd):
    """
       训练线性回归模型，并应用L2正则化
       :param lambd: 超参数，权重衰减超参数
       :return: 无
       """
    # 初始化模型参数,定义模型和损失函数,设置训练轮数和学习率
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.004
    # 创建动画器，用于绘制损失曲线
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)  # 随机梯度下降
        # 每隔5个epoch，绘制当前模型在训练集和测试集上的损失
        if (epoch + 1) % 5 == 0:
            # 在动画器中绘制训练损失和测试损失
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
        # print('w的L2范数是：', torch.norm(w).item())
    print('w的L2范数是：', torch.norm(w).item())


if __name__ == '__main__':
    train_L1(lambd=0)
    train_L1(lambd=3)
    train_L1(lambd=10)

    train_L2(lambd=0)
    train_L2(lambd=3)
    train_L2(lambd=10)
    d2l.plt.show()


