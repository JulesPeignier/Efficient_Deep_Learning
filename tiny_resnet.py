import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TinyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TinyResNet, self).__init__()
        self.in_planes = 32  # Reduced from 64

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        #print(block.expansion)
        self.linear = nn.Linear(128 * block.expansion, num_classes)

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
        out = F.avg_pool2d(out, 8)
        # print(out.shape)  # Debug print to check the shape
        out = out.view(out.size(0), -1)
        # print(out.shape)  # Debug print to check the shape after flattening
        out = self.linear(out)
        return out

# class NanoResNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(NanoResNet, self).__init__()

#         self.conv = nn.Sequential(

#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=64, momentum=0.9),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(kernel_size=2, stride=2),

#             ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),

#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=128, momentum=0.9),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(kernel_size=2, stride=2),

#             ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),

#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

#     def forward(self, x):
#         out = self.conv(x)
#         out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
#         out = self.fc(out)
#         return out


class NanoResNet2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(NanoResNet2, self).__init__()
        self.in_planes = 16  # Reduced from 64

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #print(block.expansion)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

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
        out = F.avg_pool2d(out, 8)
        # print(out.shape)  # Debug print to check the shape
        out = out.view(out.size(0), -1)
        # print(out.shape)  # Debug print to check the shape after flattening
        out = self.linear(out)
        return out
    
def TinyResNet18():
    return TinyResNet(BasicBlock, [2, 2, 2])

def MicroResNet():
    return TinyResNet(BasicBlock, [1, 1, 1])

def NanoResNet():
    return NanoResNet2(BasicBlock, [2, 2, 2])

# Example test
def test():
    net = TinyResNet18()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters in TinyResNet: {total_params}')
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

    # net = NanoResNet()
    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f'Total parameters in NanoResNet: {total_params}')
    # y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())

    net = NanoResNet()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters in NanoResNet: {total_params}')
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

    net = MicroResNet()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters in MicroResNet: {total_params}')
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()

