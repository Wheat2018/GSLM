from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.extractor import resnet50_cam

class Net(resnet50_cam.Net):
    def __init__(self, class_num=20):
        super().__init__(class_num)

        self.mapping = nn.Sequential(nn.AdaptiveAvgPool2d((32, 32)), nn.Conv2d(20, 20, 32, bias=False))
        self.newly_added.append(self.mapping)

    def get_cls(self, feature):
        x = self.mapping(feature)
        x = x.view(-1, self.class_num)
        return x
    
    def get_cam(self, feature):
        x = F.relu6(feature)
        return x

    @classmethod
    def mixin_multi_cams(cls, cams: Iterable[torch.Tensor], upsample_size):
        cams = [F.interpolate(cam, upsample_size, mode='bilinear', align_corners=False) for cam in cams]
        cams = torch.mean(torch.stack(cams, 0), 0)
        cams /= 6.0
        return cams