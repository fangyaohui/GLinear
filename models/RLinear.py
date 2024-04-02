
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        configs.rev  = True
        configs.individual  = True
        self.pred_len = configs.pred_len

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.enc_in)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(0.1)
        self.rev = RevIN(configs.enc_in) if configs.rev else None
        self.individual = configs.individual



    def forward(self, x):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred
