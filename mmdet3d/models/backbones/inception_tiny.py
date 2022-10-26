import torch
import torch.nn as nn
# from torchsummary import summary
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class InceptionForBEV(nn.Module):
    """
    残差模块
    """
    def __init__(self, in_channel, out_channel, stride=2):
        super(InceptionForBEV, self).__init__()
        # 1*1 conv
        # self.conv1 = nn.Conv2d(in_channel, out_channel[0], 1, 1)
        # 1*1 3*3 conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel[1], 1, 1),
            nn.Conv2d(out_channel[1], out_channel[1], 3, stride, 1)
        )
        # 1*1, 3*3 conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channel[1], out_channel[2], 1, 1, 1),
            nn.Conv2d(out_channel[2], out_channel[2], 5, stride, 1)
        )
        # maxpoolConv
        self.maxpoolConv = nn.Sequential(
            nn.MaxPool2d(3, 1),
            nn.Conv2d(out_channel[2], out_channel[3], 1, stride, 1)
        )

        self.bn1 = nn.BatchNorm2d(out_channel[1])
        self.bn2 = nn.BatchNorm2d(out_channel[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = []
        out3 = self.conv3(input)
        out3 = self.bn1(out3)
        out3 = self.relu(out3)

        out5 = self.conv5(out3)
        out5 = self.bn2(out5)
        out5 = self.relu(out5)
        pool = self.maxpoolConv(out5)
        # out = torch.cat([out1, out3, out5, pool], dim=1)
        output.extend([out3, out5, pool])
        # print(output.__len__())
        # print(output[0].shape)
        # print(output[1].shape)
        # print(output[2].shape)

        return output


if __name__ == '__main__':
    # net = Inception_v1()
    # if torch.cuda.is_available():
    #     summary(net.cuda(), (3, 224, 224))
    # else:
    #     summary(net, (3, 224, 224))

    inputs = torch.randn(1, 64, 224, 224)
    net = InceptionForBEV(64, [64, 128, 256, 512])
    outputs = net(inputs)
    print(outputs.__len__())
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
