import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prob_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, prob_dim):
        super(Prob_Linear, self).__init__()
        self.output = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, output_dim, 1, 1, 0, bias=True),
        )
        self.prob = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, input_dim // 2, 3, 1, 0, bias=False),
            nn.BatchNorm1d(input_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim // 2, prob_dim, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        outs = self.output(x)
        outs = outs.view(outs.size(0), 3, -1, outs.size(2))

        probs = 0.1 + F.softplus(self.prob(x))
        
        return outs, probs
