import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
from net.extractor.extractor import Extractor


class Net(Extractor):

    def __init__(self, class_num=20):
        super().__init__()

        base = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        for p in base.conv1.parameters():
            p.requires_grad = False
        for p in base.bn1.parameters():
            p.requires_grad = False

        self.stage1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                    base.layer1)
        self.stage2 = nn.Sequential(base.layer2)
        self.stage3 = nn.Sequential(base.layer3)
        self.stage4 = nn.Sequential(base.layer4)

        self.classifier = nn.Conv2d(2048, class_num, 1, bias=False)

        self.backbone = [self.stage1, self.stage2, self.stage3, self.stage4]
        self.newly_added = [self.classifier]
        self.class_num = class_num

    def trainable_parameters(self):
        return nn.ModuleList(self.backbone).parameters(), nn.ModuleList(self.newly_added).parameters()

    def get_feature(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        if self.training:
            x = x.detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = self.classifier(x)
        return x

    def get_cls(self, feature):
        x = torchutils.gap2d(feature, keepdims=True)
        x = x.view(-1, self.class_num)
        return x
    
    def get_cam(self, feature):
        x = F.relu(feature)
        return x
