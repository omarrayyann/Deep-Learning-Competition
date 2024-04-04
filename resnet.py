import torch
from torch import nn
import torch.nn.functional as F


class ResNet(nn.Module):

    def __init__(self, num_blocks):
        super().__init__()

        # 1. Initial Convolutional Layer (kernel size 3x3 with stride 1 and padding 1):
        # - Reference: https://www.nature.com/articles/nature14539 
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    32 x 32 - (3 Channels)
        # - Output:   32 x 32 - (64 Channels)
        self.inital_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()

        # 2. Squeeze and Excitation Layer:
        # - Reference: https://arxiv.org/pdf/1709.01507.pdf
        # - Input:    32 x 32 - (64 Channels)
        # - Output:   32 x 32 - (64 Channels)
        self.seblock = SE_Block(64) 

        # 3. Residual Convolutional Layers:
        # - Reference: https://arxiv.org/abs/1512.03385
        # - Input:    32 x 32 - (64 Channels)
        # - Output:   4 x 4 - (256 Channels)
        self.residual_layers = nn.ModuleList([
            ResdiaulBlock.make_batch(64, 64, num_blocks[0], stride=1),
            ResdiaulBlock.make_batch(64, 128, num_blocks[1], stride=2),
            ResdiaulBlock.make_batch(128, 256, num_blocks[2], stride=2),
        ])

        # 4. Average Pooling:
        # - Input:   4 x 4 - (256 Channels)
        # - Output:  1 x 1 - (256 Channels)
        
        # 5. Flattening:
        # - Input:   1 x 1 - (256 Channels)
        # - Output:  256 x 1


        # 6. Linear Fully Connected Layer:
        # - Input:    256 x 1
        # - Output:   10 x 1
        self.linear = nn.Linear(256, 10)    

    def forward(self, x):
        # Initial Convolutional Layer
        # 32 x 32 - (3 Channels)
        out = self.inital_conv(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.seblock(out) 
        # 32 x 32 - (64 Channels)
        for layer in self.residual_layers: 
            out = layer(out) 
        # 4 x 4 - (256 Channels)
        out = F.avg_pool2d(out, 8)
        # 1 x 1 - (256 Channels)
        out = out.view(out.size(0), -1)
        # 256 x 1
        out = self.linear(out)
        # 256 x 10
        return out
    
    
# Input Dimensions:   W x H x in_channels
# Output Dimensions:  W/stride x H/stride x out_channels
class ResdiaulBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Convolutional Layer 1:
        # - Note: Padding is added to limit the width (W) and height (H) changes to /stride
        # - Input:    W x H x in_channels
        # - Output:   W/stride x H/stride x out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # Batch Normalization Layer 1:
        # - Input:    Unchanged
        # - Output:   Unchanged
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # ReLU Activation Layer 1:
        # - Input:    Unchanged
        # - Output:   Unchanged
        self.relu1 = nn.ReLU()

        # Convolutional Layer 2 (kernel size 3x3 with stride 1 and padding 1):
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    W x H x in_channels
        # - Output:   Unchanged
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        
        # Batch Normalization Layer 2:
        # - Input:    Unchanged
        # - Output:   Unchanged
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # ReLU Activation Layer 2:
        # - Input:    Unchanged
        # - Output:   Unchanged
        self.relu2 = nn.ReLU()

        # Convolutional Layer with a Batch Normalization to match the input and output dimensions
        # - Input:    W x H x in_channels
        # - Output:   W/stride x H/stride x out_channels
        self.shortcut = nn.Sequential()
        if in_channels!=out_channels or stride!=1:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(self.in_channels,self.out_channels,kernel_size=1,stride=self.stride,bias=False),
                    nn.BatchNorm2d(self.out_channels)
                )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x = self.shortcut(x)
        out = x + out
        out = F.relu(out)
        return out

    @staticmethod
    def make_batch(in_channels, out_channels, batch_size, stride):
        layers = []
        layers.append(ResdiaulBlock(in_channels, out_channels, stride=stride))
        for _ in range(1,batch_size):
            layers.append(ResdiaulBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(c, c // r, 1, 1, groups=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(c // r, c, 1, 1, groups=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y
    