import torch
from torch import nn
import torch.nn.functional as F

# Input Dimensions:   W x H x in_channels
# Output Dimensions:  W/stride x H/stride x out_channels
class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Convolutional Layer 1 (kernel size 3x3 with stride 1 and padding 1):
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    W x H x in_channels
        # - Output:   W/stride x H/stride x out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # Batch Normalization Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # Convolutional Layer 2 (kernel size 3x3 with stride 1 and padding 1):
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    W x H x in_channels
        # - Output:   W x H x out_channels
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        
        # Batch Normalization Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        
        if self.out_channels != self.in_channels or self.stride!=1:
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

def make_residual_layer(num_blocks, in_channels, out_channels, stride):
    layers = []
    layers.append(BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
    for _ in range(num_blocks):
        layers.append(BasicBlock(in_channels=out_channels, out_channels=out_channels, stride=1))
    return nn.Sequential(*layers)

class SEBlock(nn.Module):

    def __init__(self,
                 channels,
                 reduction=16):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_cannels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=True)
        self.activ = nn.ReLU(inplace=True) 

        self.conv2 = nn.Conv2d(
            in_channels=mid_cannels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=True)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class ResNet(nn.Module):

    def __init__(self, num_blocks):
        super().__init__()

        self.avg_pool_kernel_size = 8

        # Initial Convolutional Layer (kernel size 3x3 with stride 1 and padding 1):
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    32 x 32 x 3
        # - Output:   32 x 32 x 64
        self.inital_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.seblock = SEBlock(channels=64) 

        self.residual_layers = [] 
        self.residual_layers.append(self._make_layer(BasicBlock, 64, 64, num_blocks[0], stride=1)) 
        self.residual_layers.append(self._make_layer(BasicBlock, 64, 128, num_blocks[1], stride=2)) 
        self.residual_layers.append(self._make_layer(BasicBlock, 128, 256, num_blocks[2], stride=2)) 
        self.residual_layers = nn.ModuleList(self.residual_layers)

        self.linear = nn.Linear(256, 10)



    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
        

    def forward(self, x):

        out = F.relu(self.bn1(self.inital_conv(x)))

        out = self.seblock(out) 
        for layer in self.residual_layers: 
            out = layer(out) 

        out = F.avg_pool2d(out, self.avg_pool_kernel_size)

        out = out.view(out.size(0), -1)
        
        out = self.linear(out)

        return out