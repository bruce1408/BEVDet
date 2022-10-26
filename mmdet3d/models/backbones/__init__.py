# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt, MobileNetV2
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .pointnet2_sa_msg import PointNet2SAMSG
from .pointnet2_sa_ssg import PointNet2SASSG
from .second import SECOND
from .resnet import ResNetForBEVDet
from .swin import SwinTransformer
from .inception_tiny import InceptionForBEV  # 新加的inception网络
# from torchvision.models.vgg import vgg11_bn

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'PointNet2SASSG', 'PointNet2SAMSG', 'MultiBackbone',
    'ResNetForBEVDet', 'SwinTransformer', 'InceptionForBEV'
]
