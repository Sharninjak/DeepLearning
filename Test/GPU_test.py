import torch
import os
from torch import nn

print('torch.__version__:', torch.__version__)
print('torch.cuda.is_available:', torch.cuda.is_available())
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
print('CUDA_VISIBLE_DEVICES:', cuda_visible_devices)
print('torch.cuda.current_device():', torch.cuda.current_device())
print('torch.cuda.device_count():', torch.cuda.device_count())
print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


def __init__(self, data_path, tokenizer=lambda x: x.split(), vocab=None):
    """
    Initialize the dataset.
    :param self:
    :param data_path:
    :param tokenizer:
    :param vocab:
    :return:
    """

def try_gpu(i=0):

    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(try_gpu(), try_gpu(0), try_gpu(1), try_all_gpus())

x = torch.tensor([1,2,3])
print(x.device)

y = torch.rand(2, 3, device=try_gpu(0))
print(y)  # device='cuda:0'

z = torch.rand(2, 3, device=try_gpu(1))
print(z)  # no cuda

a = y.cuda(0)
print(x)
print(a)  # device='cuda:1'

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
# print(net(x))  # false, x is on cpu
print(net(y))  # true, y is on gpu
