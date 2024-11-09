import torch

print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.cuda.device_count())  # 检查可用GPU数量
