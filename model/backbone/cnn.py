import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

class VoiceBlock(nn.Module):
    def __init__(self, planes):
        super(VoiceBlock, self).__init__()
        self.conv1 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes, affine=True)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class VoiceFeatNet(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, 
                 cnn_channel, feat_dim,
                 kernel_size=3, stride=2, padding=1, alpha=1.5):
        super(VoiceFeatNet, self).__init__()
        # waveform to melspec
        self.melspec_extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft,
                win_length=400, hop_length=160, n_mels=n_mels)
        #self.spec_extractor = torchaudio.transforms.Spectrogram(
        #        n_fft=n_fft, win_length=400, hop_length=160)
        self.mvn_w = 101

        # melspec to embedding
        cnn_channels = [int(alpha**i*cnn_channel) for i in range(4)]
        self.model = nn.Sequential(
             nn.Conv1d(n_mels, cnn_channels[0],
                 kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(cnn_channels[0], affine=True),
             nn.ReLU(inplace=True),
             VoiceBlock(cnn_channels[0]),
             nn.Conv1d(cnn_channels[0], cnn_channels[1],
                 kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(cnn_channels[1], affine=True),
             nn.ReLU(inplace=True),
             #nn.Dropout(p=0.5),
             VoiceBlock(cnn_channels[1]),
             nn.Conv1d(cnn_channels[1], cnn_channels[2],
                 kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(cnn_channels[2], affine=True),
             nn.ReLU(inplace=True),
             #nn.Dropout(p=0.5),
             VoiceBlock(cnn_channels[2]),
             nn.Conv1d(cnn_channels[2], cnn_channels[3],
                 kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(cnn_channels[3], affine=True),
             nn.ReLU(inplace=True),
             #nn.Dropout(p=0.5),
             VoiceBlock(cnn_channels[3]),
             nn.Conv1d(cnn_channels[3], feat_dim,
                 kernel_size, stride, padding, bias=True),
        )

        # init weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''


    def forward(self, x):
        h_win = self.mvn_w // 2
        x = x[:, 1:] - 0.97 * x[:, :-1]
        x = self.melspec_extractor(x)
        #x = self.spec_extractor(x)
        x = torch.log(x + 10e-6) / 2.3

        x = torch.unsqueeze(x, 1)
        x = (x[:, :, :, h_win:-h_win]
           - F.avg_pool2d(x, (1, self.mvn_w), stride=1))
        x = torch.squeeze(x, 1)

        x = self.model(x)
        return x

