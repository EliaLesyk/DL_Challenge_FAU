import torch
import torch.nn.functional as F

# definition of a ResBlock used by the ResNet
# one block is basically treated as its own network 
class ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        # define building blocks of a single ResBlock
        self._conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride))
        self._norm1 = torch.nn.BatchNorm2d(out_channels)
        self._conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1))
        self._norm2 = torch.nn.BatchNorm2d(out_channels)
        # downsampling layer
        size = 5 + 2 * (stride - 1)
        self._conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(size, size), stride=(stride, stride))
        self._norm3 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # first convolutional pass
        out_tensor = self._conv1(x)
        out_tensor = self._norm1(out_tensor)
        out_tensor = F.relu(out_tensor, inplace=True)
        # second convolutional pass
        out_tensor = self._conv2(out_tensor)
        out_tensor = self._norm2(out_tensor)
        out_tensor = F.relu(out_tensor, inplace=True)
        # 1D-Conv pass to downsample input
        input_tensor = self._conv3(x)
        input_tensor = self._norm3(input_tensor)
        # add input to output
        out_tensor = out_tensor + input_tensor
        out_tensor = F.relu(out_tensor, inplace=True)
        return out_tensor

class ResNet(torch.nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        # define building blocks of the network
        # convolutional pass
        self._conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2))
        self._norm1 = torch.nn.BatchNorm2d(64)
        self._maxp1 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # add them as child module so they are automatically set to train/eval mode
        self._block1 = ResBlock(64, 64, 1)
        self.add_module("block1", self._block1)
        self._block2 = ResBlock(64, 128, 2)
        self.add_module("block2", self._block2)
        self._block3 = ResBlock(128, 256, 2)
        self.add_module("block3", self._block3)
        self._block4 = ResBlock(256, 512, 2)
        self.add_module("block4", self._block4)
        # fully connected pass
        self._fc1 = torch.nn.Linear(512*4*4, 2)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 2D-Conv pass
        out_tensor = self._conv1(x)
        out_tensor = self._norm1(out_tensor)
        out_tensor = F.relu(out_tensor, inplace=True)
        out_tensor = self._maxp1(out_tensor)
        # ResBlock pass
        out_tensor = self._block1(out_tensor)
        out_tensor = self._block2(out_tensor)
        out_tensor = self._block3(out_tensor)
        out_tensor = self._block4(out_tensor)
        out_tensor = out_tensor.flatten(1)
        # Fully Connected pass
        out_tensor = self._fc1(out_tensor)
        out_tensor = self._sigmoid(out_tensor)
        return out_tensor





