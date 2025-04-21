"""
@author: S.Tahir.H.Rizvi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .RevIN import RevIN

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in

        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)

        self.revin_layer = RevIN(self.in_num_features)

        # ================== 可学习的频率滤波器参数 ==================
        self.scale = 0.02  # 滤波器参数初始化缩放因子（防止梯度爆炸）
        self.embed_size = self.seq_len  # 滤波器带宽等于输入长度
        # 可学习的频率滤波器参数（对应公式8）
        # 使用通道共享策略（1 x L 形状），即所有变量共享同一滤波器
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

    def forward(self, x):

        # 输入序列长度 L 论文中的lookback Window
        # 预测长度 τ
        # 特征数目 N

        # B L N
        # x shape is (16,336,7)
        # 分别为batch 输入步长 输出指标数目

        # ================== 预处理阶段 ==================
        x = self.revin_layer(x, 'norm')

        # ================== GLinear ==================
        # 维度置换 [B, N, L]（N为变量数，对应论文的channel-independence策略）
        x3 = x.permute(0,2,1)
        # x3 处理结束之后(16,7,336)

        # 应用循环卷积（论文4.2节核心操作）
        x3 = self.circular_convolution(x3, self.w.to(x.device))  # [B, N, L]

        x3 = self.Linear(x3)
        # x3 经过一个全连接层处理结束之后依旧为(16,7,336)
        x3 = self.GeLU(x3)
        x3 = self.Hidden1(x3)
        # x3 经过一个隐藏层处理之后为（16,7,24）

        # ================== 后处理阶段 ==================
        x3 = x3.permute(0,2,1)
        # x3 处理之后变为（16,24,7）
        x3 = self.revin_layer(x3, 'denorm')
        # x3 处理之后变为（16,24,7）

        x = x3
        return x # [Batch, Output length, Channel]

    def circular_convolution(self, x, w):
        """论文公式1实现的循环卷积操作
        对应论文4.2节的频域滤波过程（公式5）

        参数：
            x: 输入信号 [B, N, L]
            w: 滤波器参数 [1, L]
        返回：
            [B, N, L] 滤波后的信号
        """
        # 输入信号FFT（公式8）
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # 转换为频域 [B, N, L//2+1]

        # 滤波器参数FFT（对应论文中的H_φ）
        w = torch.fft.rfft(w, dim=1, norm='ortho')  # [1, L//2+1]

        # 频域点乘滤波（对应公式8中的逐元素乘积）
        y = x * w

        # 逆FFT恢复时域信号
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out