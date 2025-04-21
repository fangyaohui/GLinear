"""
@author: S.Tahir.H.Rizvi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .RevIN import RevIN

class GLinearModel(nn.Module):
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

    def forward(self, x):

        # x shape is (16,336,7)
        # 分别为batch 输入步长 输出指标数目

        x = self.revin_layer(x, 'norm')
        x3 = x.permute(0,2,1)
        # x3 处理结束之后(16,7,336)

        x3 = self.Linear(x3)
        # x3 经过一个全连接层处理结束之后依旧为(16,7,336)
        x3 = self.GeLU(x3)
        x3 = self.Hidden1(x3)
        # x3 经过一个隐藏层处理之后为（16,7,24）

        x3 = x3.permute(0,2,1)
        # x3 处理之后变为（16,24,7）
        x3 = self.revin_layer(x3, 'denorm')
        # x3 处理之后变为（16,24,7）

        x = x3
        return x # [Batch, Output length, Channel]
