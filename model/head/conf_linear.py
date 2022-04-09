import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conf_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, conf_dim):
        super(Conf_Linear, self).__init__()
        self.output = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1, 1, 0, bias=True),
        )
        self.conf = nn.Sequential(
            nn.Conv1d(input_dim, conf_dim, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        outs = self.output(x)
        confs = F.softplus(self.conf(x)) / 0.6931

        return outs, confs
