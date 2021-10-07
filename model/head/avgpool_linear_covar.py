import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool_Linear_Covar(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AvgPool_Linear_Covar, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
        )
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

        #self.output.apply(init_weights)

    def forward(self, x):
        batch_size = x.size(0)
        num_frame = x.size(2)
        x = F.avg_pool1d(x, num_frame, stride=1)
        x = x.view(batch_size, -1)
        outs = self.output(x)

        return outs
