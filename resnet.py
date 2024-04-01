import torch
from torch import nn
import torch.nn.functional as F

# Input Dimensions:   W x H x in_channels
# Output Dimensions:  W/stride x H/stride x out_channels
class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, combine_function=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.combine_function = combine_function
        # Convolutional Layer 1 (kernel size 3x3 with stride 1 and padding 1):
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    W x H x in_channels
        # - Output:   W/stride x H/stride x out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
        # Batch Normalization Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        # ReLU Activation Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.relu1 = nn.ReLU()
        # Convolutional Layer 2 (kernel size 3x3 with stride 1 and padding 1):
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    W x H x in_channels
        # - Output:   W x H x out_channels
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        # Batch Normalization Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        # ReLU Activation Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.relu2 = nn.ReLU()
        # A function that combines the input and output before the activation function
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.combine_function is not None:
            out = self.combine_function(x, out, self.in_channels, self.out_channels, self.stride)
        else:
            out = x + out

        out = self.relu2(out)

        return out

def make_residual_layer(num_blocks, in_channels, out_channels, stride, combine_function=None):
    layers = []
    layers.append(BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, combine_function=combine_function))
    for _ in range(num_blocks):
        layers.append(BasicBlock(in_channels=out_channels, out_channels=out_channels, stride=1, combine_function=combine_function))
    return nn.Sequential(*layers)


class ResNet(nn.Module):

    def __init__(self, num_blocks, combine_function):

        super().__init__()
        # Initial Convolutional Layer (kernel size 3x3 with stride 1 and padding 1):
        # - Note: Padding is added to retain the width (W) and height (H)
        # - Input:    32 x 32 x 3
        # - Output:   32 x 32 x 16
        self.inital_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.layer1 = make_residual_layer(
            in_channels=16,
            out_channels=16,
            num_blocks=num_blocks[0],
            combine_function=None,
            stride=1
        )
        # Second Residual Layer
        # - Input:    32 x 32 x 16
        # - Output:   16 x 16 x 32
        self.layer2 = make_residual_layer(
            in_channels=16,
            out_channels=32,
            num_blocks=num_blocks[1],
            combine_function=combine_function,
            stride=2,
        )
        # Third Residual Layer
        # - Input:    16 16 8 x 32
        # - Output:   8 x 8 x 64
        self.layer3 = make_residual_layer(
            in_channels=32,
            out_channels=64,
            num_blocks=num_blocks[2],
            combine_function=combine_function,
            stride=2,
        )
        # Average Pooling
        # - Input:   8 x 8 x 64
        # - Output:  1 x 1 x 64
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8)
        # ReLU Activation Layer
        # - Input:   4 x 4 x 64
        # - Output:  1 x 1 x 64
        self.relu = nn.ReLU()
        # Linear Layer
        # - Input:   1 x 1 x 64
        # - Output:  1 x 1 x 10
        self.linear = nn.Linear(64, 10)
        

    def forward(self, x):
        out = self.inital_conv(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.relu(out)
        out = self.linear(out)
        return out