import torch

print(torch.__version__)          # 应显示 2.x.x（如 2.5.0）
print(torch.cuda.is_available())  # 必须返回 True（表示 GPU 可用）
print(torch.version.cuda)         # 应显示 12.4（CUDA 版本）
