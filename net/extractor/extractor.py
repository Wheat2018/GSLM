from typing import Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def forward(self, x):
        feature = self.get_feature(x)
        return {'cls': self.get_cls(feature), 'cam': self.get_cam(feature)}
    
    def get_feature(self, x: torch.Tensor):
        raise NotImplementedError

    def get_cls(self, x: torch.Tensor):
        raise NotImplementedError
    
    def inference(self, img: torch.Tensor, scales: Tuple[float]=(1.0,), flip=True, upsample_size=None):
        if upsample_size is None:
            upsample_size = img.shape[-2:]

        if img.ndim == 3:
            img.unsqueeze_(0)
        assert img.ndim == 4
        N, C, H, W = img.shape

        images = self.multiscale_images(img, scales)
        if flip:
            images = [torch.cat((image, image.flip(-1))) for image in images]
        cams = []
        for image in images:
            out = self(image)['cam']
            cams.append(out[:N])
            if flip:
                cams.append(out[N:].flip(-1))
        return self.mixin_multi_cams(cams, upsample_size)
    
    @classmethod
    def mixin_multi_cams(cls, cams: Iterable[torch.Tensor], upsample_size):
        assert all(x.ndim == 4 for x in cams)
        cams = [F.interpolate(cam, upsample_size, mode='bilinear', align_corners=False) for cam in cams]
        cams = torch.sum(torch.stack(cams, 0), 0)
        cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5
        return cams


    @classmethod
    def multiscale_images(cls, img: torch.Tensor, scales: Tuple[float]):
        return [F.interpolate(img, scale_factor=scale, mode='nearest', recompute_scale_factor=True) for scale in scales]

    def __repr__(self):
        return repr(self.__class__)