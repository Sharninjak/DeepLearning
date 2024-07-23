import torch
import os


cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
# 打印值
if cuda_visible_devices is not None:
    print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
else:
    print("CUDA_VISIBLE_DEVICES is not set.")
print(torch.cuda.current_device())
print(torch.cuda.device_count())

