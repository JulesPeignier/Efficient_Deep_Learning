import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Define depthwise separable convolution
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# Define the basic block using depthwise separable convolution
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = DepthwiseSeparableConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DepthwiseSeparableConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define Tiny ResNet Module
class DSC_TinyResNet_Module(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(DSC_TinyResNet_Module, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.linear = nn.Linear(128 * block.expansion * 4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Define Nano ResNet Module
class DSC_NanoResNet_Module(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(DSC_NanoResNet_Module, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion * 4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Helper functions to create model instances
def DSC_TinyResNet():
    return DSC_TinyResNet_Module(BasicBlock, [2, 2, 2])

def DSC_MicroResNet():
    return DSC_TinyResNet_Module(BasicBlock, [1, 1, 1])

def DSC_NanoResNet():
    return DSC_NanoResNet_Module(BasicBlock, [2, 2, 2])

# Example test
def test():
    net = DSC_TinyResNet()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters in DSC_TinyResNet: {total_params}')
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

    net = DSC_MicroResNet()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters in DSC_MicroResNet: {total_params}')
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

    net = DSC_NanoResNet()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters in DSC_NanoResNet: {total_params}')
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
