import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prob_Deconv(nn.Module):
    def __init__(self, input_dim, deconv_channel, UV_path, alpha=1.5):
        super(Prob_Deconv, self).__init__()
        # get position map
        deconv_channels = [int(alpha**i*deconv_channel) for i in range(5)]
        self.output = nn.Sequential(
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
        self.prob = nn.Sequential(
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
            nn.ConvTranspose2d(deconv_channels[4], 1, 1, 1, 0, bias=True),
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
        batch_size = x.size(0)
        num_channel = x.size(1)
        num_frame = x.size(2)
        ''' x: [batch_size x num_channel x num_frame] 
        '''
        x = torch.transpose(x, 1, 2)
        ''' x: [batch_size x num_frame x num_channel] 
        '''
        x = x.view(-1, num_channel, 1, 1)
        ''' x: [batch_size * num_frame x num_channel] 
        '''
        
        outs = self.output(x)
        ''' outs: [batch_size * num_frame x 3 x height x width] 
        '''
        outs = outs.view(batch_size * num_frame, 3, -1)
        ''' outs: [batch_size * num_frame x 3 x height * width] 
        '''
        outs = (outs[:, :, self.indices[0]] * self.weights[:, :, 0]
                + outs[:, :, self.indices[1]] * self.weights[:, :, 1]
                + outs[:, :, self.indices[2]] * self.weights[:, :, 2]
                + outs[:, :, self.indices[3]] * self.weights[:, :, 3])
        ''' outs: [batch_size * num_frame x 3 x num_vertex] 
        '''
        outs = outs.view(batch_size, num_frame, 3, -1)
        ''' outs: [batch_size x num_frame x 3 x num_vertex] 
        '''
        outs = torch.transpose(outs, 1, 2)
        outs = torch.transpose(outs, 2, 3)
        ''' outs: [batch_size x 3 x num_vertex x num_frame]
        '''

        probs = self.prob(x)
        ''' probs: [batch_size * num_frame, 1, height, width]
        '''
        probs = probs.view(batch_size * num_frame, 1, -1)
        probs = (probs[:, :, self.indices[0]] * self.weights[:, :, 0]
                 + probs[:, :, self.indices[1]] * self.weights[:, :, 1]
                 + probs[:, :, self.indices[2]] * self.weights[:, :, 2]
                 + probs[:, :, self.indices[3]] * self.weights[:, :, 3])
        probs = probs.view(batch_size, num_frame, -1)
        probs = torch.transpose(probs, 1, 2)
        ''' probs: [batch_size x num_vertex x num_frame]
        '''

        return outs, probs
