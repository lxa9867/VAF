import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool_Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AvgPool_Linear, self).__init__()
        self.output = nn.Sequential(
            #nn.Linear(input_dim, input_dim // 2, bias=False),
            #nn.BatchNorm1d(input_dim // 2, affine=True),
            #nn.ReLU(inplace=True),
            #nn.Linear(input_dim // 2, input_dim // 2, bias=False),
            #nn.BatchNorm1d(input_dim // 2, affine=True),
            #nn.ReLU(inplace=True),
            #nn.Linear(input_dim // 2, output_dim, bias=True),
            nn.Linear(input_dim, output_dim, bias=True),
        )
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, mean=0.0, std=0.05)

        self.output.apply(init_weights)



    def forward(self, x):
        batch_size = x.size(0)
        num_frame = x.size(2)
        x = F.avg_pool1d(x, num_frame, stride=1)
        x = x.view(batch_size, -1)
        outs = self.output(x)
        outs = outs.view(batch_size, 3, -1, 1)

        confs = torch.ones_like(outs[:, 0:1, :, :])

        return outs, confs
