import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = pool
        if pool:
            self.maxpool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = ConvBlock(2, 16, 6)
        self.conv2 = ConvBlock(16, 24, 6, pool=True)
        self.conv3 = ConvBlock(24, 32, 6, pool=True)
        self.conv4 = ConvBlock(32, 48, 6, pool=True)
        self.conv5 = ConvBlock(48, 64, 6, pool=True)
        self.conv6 = ConvBlock(64, 128, 6, pool=False)
        self.conv7 = ConvBlock(128, 256, 6, pool=False)

        self.linear1 = nn.LazyLinear(128)
        self.linear2 = nn.LazyLinear(6)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        output = self.linear2(x)

        return output
