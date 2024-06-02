import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import imutils
from net.extractor import resnet50_cam

def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Net(resnet50_cam.Net):
    def __init__(self, class_num=20):
        super().__init__(class_num)

        self.decode2 = Decoder(2048, 128 + 512, 128)
        self.decode1 = Decoder(128, 64 + 256, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )

        self.classifier = nn.Conv2d(64, 20, 1, bias=False)
        
        self.newly_added = [self.decode2, self.decode1, self.decode0, self.classifier]
        _initialize_weights(self.newly_added)

    def get_feature(self, x):
        size = imutils.get_strided_up_size(x.shape[-2:], 16)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size, mode='bilinear', align_corners=False)
        x4 = self.stage1(x)  # 256, 64, 64
        x8 = self.stage2(x4).detach()   # 512, 32, 32

        if self.training:
            x8 = x8.detach()

        x16 = self.stage3(x8)  # 1024, 16, 16
        x16 = self.stage4(x16)  # 2048, 16, 16

        d2 = self.decode2(x16, x8)  # 128,32,32
        d1 = self.decode1(d2, x4)  # 64,64,64
        d0 = self.decode0(d1)  # 64,128,128

        x = self.classifier(d0)
        return x

    def get_cam(self, feature):
        x = F.relu6(feature)
        return x

