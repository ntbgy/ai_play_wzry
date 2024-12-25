import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 打印出可用的 CUDA 设备数量
    print("可用的 CUDA 设备数量:", torch.cuda.device_count())
    # 遍历每个可用的 CUDA 设备，并打印出设备编号和设备名称
    for device_idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{device_idx}")
        print(f"设备编号: {device_idx}, 设备名称: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA 不可用，可能没有安装 NVIDIA GPU 或者 CUDA 驱动")
