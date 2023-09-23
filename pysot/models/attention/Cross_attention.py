import torch.nn as nn
import torch
import math

class ATTBlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ATTBlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1a]
        v_avg = self.avg_pool(x)
        v_max = self.max_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v_avg = self.conv(v_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v_max = self.conv(v_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v_avg + v_max)

        return v