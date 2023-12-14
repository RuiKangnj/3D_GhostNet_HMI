import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv3d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size//2, groups=init_channels, bias=False),
            nn.BatchNorm3d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :, :]

class ChannelAttentionModule3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule3D(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule3D, self).__init__()
        self.conv3d = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv3d(out)
        return self.sigmoid(out)

class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttentionModule3D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule3D()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(256, 4)  # 假设4个分类

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)  # 扁平化
        x = self.layer5(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(32, 4)  # 假设4个分类

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 获取序列的最后一个输出
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
