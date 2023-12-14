import torch
import torch.nn as nn
from nn.modules import GhostModule,CBAM3D
import torch.nn.functional as F
class GhostNet(nn.Module):
    def __init__(self, num_classes=4):
        super(GhostNet, self).__init__()
        # 第一层卷积 初始化处理256的通道
        self.conv1 = nn.Conv3d(256, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # 使用 Ghost 模块
        self.ghost_module1 = GhostModule(64, 128)
        self.ghost_module2 = GhostModule(128, 256)
        self.ghost_module3 = GhostModule(256, 512)

        # CBAM 3D 模块
        self.cbam = CBAM3D(in_channels=512)

        # 自适应平均池化到 1x1x1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类器
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.ghost_module1(x)
        x = self.ghost_module2(x)
        x = self.ghost_module3(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x