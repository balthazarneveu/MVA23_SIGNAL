import torch.nn as nn
from collections import namedtuple


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


class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv1d(2, self.in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(
            block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(
            block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(
            block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, self.expansion * out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm1d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


def get_resnet(output_dim=6):
    ResNetConfig = namedtuple(
        'ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])
    model_res = ResNet(resnet50_config, output_dim)
    return model_res


class StackedGRUModel(nn.Module):
    def __init__(self, num_classes, input_size=2, hidden_size=64, num_layers=2):
        super(StackedGRUModel, self).__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.batch_norm_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Permute dimensions to match GRU input shape
        x = x.permute(0, 2, 1)

        # GRU layer
        _, hn = self.gru(x)

        # Take the hidden state from the last time step
        x = hn[-1, :, :]

        # Fully connected layers
        x = nn.functional.relu(self.batch_norm_fc(self.fc1(x)))
        x = self.fc2(x)

        return nn.functional.log_softmax(x, dim=1)


class PhysicienModel(nn.Module):
    def __init__(self, num_classes, num_input_channels=2):
        super().__init__()

        # 3 couches convolutives pour l'extraction de features temporelles
        self.conv1 = nn.Conv1d(num_input_channels, 32,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

        # 2 couches de contraction de dimension temporelle avec différents types de pooling
        self.pool1 = nn.AvgPool1d(kernel_size=64, stride=64, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=32, stride=32, padding=0)

        # Couche dense pour la classification
        self.lin1 = nn.Linear(128, num_classes)

        # Couche softmax en sortie
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # Extraction de features temporelles avec des couches convolutives
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        # Contraction de dimension temporelle avec deux types de pooling successifs
        x = self.pool1(x)
        x = self.pool2(x)

        # On redimensionne
        x = x.view(-1, 128)

        # Passage à travers la couches dense pour la classification
        x = self.lin1(x)

        # Passage à travers la couche softmax en sortie
        x = self.softmax(x)

        return x
