import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool_Deconv(nn.Module):
    def __init__(self, input_dim, deconv_channel, UV_path, alpha=1.5):
        super(AvgPool_Deconv, self).__init__()
        # get position map
        deconv_channels = [int(alpha**i*deconv_channel) for i in range(5)]
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(input_dim, deconv_channels[0], 4, 1, 0, bias=True),
            nn.BatchNorm2d(deconv_channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(deconv_channels[0], deconv_channels[1], 4, 2, 1, bias=True),
            nn.BatchNorm2d(deconv_channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(deconv_channels[1], deconv_channels[2], 4, 2, 1, bias=True),
            nn.BatchNorm2d(deconv_channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(deconv_channels[2], deconv_channels[3], 4, 2, 1, bias=True),
            #nn.BatchNorm2d(deconv_channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(deconv_channels[3], deconv_channels[4], 4, 2, 1, bias=True),
            #nn.BatchNorm2d(deconv_channels[4], affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(deconv_channels[4], 3, 1, 1, 0, bias=True),
        )

        # get indices (Nx4) and weights (Nx4)
        size = 2**len(deconv_channels)
        UV = np.loadtxt(UV_path).astype(np.float32).T
        UV = UV * (size - 1.)
        UV_low = np.floor(UV)
        UV_upp = np.ceil(UV)
        indices = np.array([
            UV_low[1] * size + UV_low[0],
            UV_low[1] * size + UV_upp[0],
            UV_upp[1] * size + UV_low[0],
            UV_upp[1] * size + UV_upp[0]], dtype=np.int64)

        weights = np.array([
            (UV_upp[0] - UV[0]) * (UV_upp[1] - UV[1]),
            (UV[0] - UV_low[0]) * (UV_upp[1] - UV[1]),
            (UV_upp[0] - UV[0]) * (UV[1] - UV_low[1]),
            (UV[0] - UV_low[0]) * (UV[1] - UV_low[1])])
        weights = weights.reshape([1, 1, 4, -1])

        self.register_buffer('indices', torch.from_numpy(indices))
        self.register_buffer('weights', torch.from_numpy(weights))


    def forward(self, x):
        x = F.avg_pool1d(x, x.size(2), stride=1)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv(x)

        x = x.view(x.size(0), 3, -1)
        outs = (x[:, :, self.indices[0]] * self.weights[:, :, 0]
                + x[:, :, self.indices[1]] * self.weights[:, :, 1]
                + x[:, :, self.indices[2]] * self.weights[:, :, 2]
                + x[:, :, self.indices[3]] * self.weights[:, :, 3])
        outs = outs.view(outs.size(0), 3, -1, 1)
        probs = torch.ones_like(outs[:, 0, :, :])

        return outs, probs
