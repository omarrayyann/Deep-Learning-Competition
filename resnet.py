import torch
from torch import nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super.__init__(self, ResidualBlock)
        layer_1 = BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=2, combine_function=None)


# Input Dimensions:   W x H x in_channels
# Output Dimensions:  W/stride x H/stride x out_channels
class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, combine_function=None):
        super().__init__()
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
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        # Batch Normalization Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        # ReLU Activation Layer:
        # - Input:    W x H x out_channels
        # - Output:   W x H x out_channels
        self.relu2 = nn.ReLU()
        # A function that combines the input and output before the activation function
        self.combine_function = combine_function
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.combine_function is not None:
            out = self.combine_function(x, out)
        else:
            out = x + out
            
        out = self.relu2(out)

        return out
