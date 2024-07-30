from cmath import exp
import torch

x = exp(2)
y = exp(1)
z = exp(0.1)
w = x + y + z
print(x/w, y/w, z/w)

a = torch.tensor([2.0, 1.0, 0.1])

softmax_a = torch.softmax(a, dim=0)
print(softmax_a)
