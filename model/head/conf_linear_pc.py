import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conf_Linear_PC(nn.Module):
    def __init__(self, input_dim, output_dim, conf_dim):
        super(Conf_Linear_PC, self).__init__()
        '''
        self.output = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, output_dim, 1, 1, 0, bias=True),
        )
        self.conf = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, conf_dim, 1, 1, 0, bias=True),
        )
        '''
        self.output = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1, 1, 0, bias=True),
        )
        self.conf = nn.Sequential(
            nn.Conv1d(input_dim, conf_dim, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        outs = self.output(x)
        confs = 0.1 + F.softplus(self.conf(x))

        return outs, confs