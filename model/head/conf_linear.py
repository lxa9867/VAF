import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conf_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, conf_dim):
        super(Conf_Linear, self).__init__()
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
        outs = outs.view(outs.size(0), 3, -1, outs.size(2))
        confs = 0.1 + F.softplus(self.conf(x))
        confs = confs.view(confs.size(0), 1, -1, confs.size(2))

        return outs, confs
