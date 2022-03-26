import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool_Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AvgPool_Linear, self).__init__()
        self.output = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        outs = self.output(x)
        confs = outs * 0. + 1.

        return outs, confs
