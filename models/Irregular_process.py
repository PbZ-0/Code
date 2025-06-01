import torch.nn as nn
from Config import config
import torch.nn.functional as F
import torch

class Irregular_process_lenth(nn.Module):
    def __init__(self, config, input_dim):
        super(Irregular_process_lenth, self).__init__()
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, config.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(config.hid_dim, config.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(config.hid_dim, input_dim * config.hid_dim, bias=True))
        self.hid_dim = config.hid_dim

        self.T_bias = nn.Parameter(torch.zeros(1, config.hid_dim))

    def forward(self, X):
        B, C, L = X.shape
        Filter = self.Filter_Generators(X)
        Filter_channelnorm = F.softmax(Filter, dim=-2)
        Filter_channelnorm = Filter_channelnorm.view(B, C, self.hid_dim, -1)
        X_broad = X.unsqueeze(dim=-2).repeat(1, 1, self.hid_dim, 1)
        X_out = torch.sum(X_broad * Filter_channelnorm, dim=-1)
        output = torch.relu(X_out + self.T_bias)
        if C > 1:
            XC_out = F.softmax(X_out, dim=-2)
            output = torch.sum(X_out * XC_out, dim=-2)
            output = output.unsqueeze(1)
            output = torch.relu(output + self.T_bias)
        else:
            output = output

        return output

