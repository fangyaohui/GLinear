"""
@author: S.Tahir.H.Rizvi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = 1

        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)
##############################################

    def forward(self, x):

			##############################################
        x3 = x.permute(0,2,1)

        x3 = self.Linear(x3)
        x3 = self.GeLU(x3)
        x3 = self.Hidden1(x3)

        x3 = x3.permute(0,2,1)
			##############################################



        x = x3
        return x # [Batch, Output length, Channel]
