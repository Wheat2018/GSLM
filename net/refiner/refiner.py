from typing import Tuple, Union
import torch
import numpy as np
from misc import imutils, indexing
from net.refiner.resnet50_irn import EdgeDisplacement
from voc12.dataloader import TorchvisionNormalize
import torch.nn.functional as F

class Refiner:
    def __call__(self, img: np.ndarray, cams: np.ndarray, threshold: Union[float, Tuple[float]], process_id=0) -> np.ndarray:
        if isinstance(threshold, tuple):
            return tuple(self.__call__(img, cams, th, process_id) for th in threshold)

        assert img.ndim == cams.ndim == 3
        assert img.dtype == np.uint8

        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
        return cams.argmax(0)
    
    def process_limit(self):
        return 0xffffffff

NoneRefiner = Refiner

class DenseCRF(Refiner):
    def __init__(self) -> None:
        super().__init__()
        self.CRF_PARAMS = {"g_sxy": 3, "g_com": 10, "b_sxy": 50, "b_rgb": 5, "b_com": 10, "t": 10}

    def __call__(self, img: np.ndarray, cams: np.ndarray, threshold: Union[float, Tuple[float]], process_id=0) -> np.ndarray:
        if isinstance(threshold, tuple):
            return tuple(self.__call__(img, cams, th, process_id) for th in threshold)

        seg = super().__call__(img, cams, threshold)

        return imutils.crf_inference_label(img, seg, n_labels=cams.shape[0] + 1, **self.CRF_PARAMS)

class IRN(Refiner):
    def __init__(self, pth='sess/res50_irn.pth', **kwargs) -> None:
        super().__init__()
        self.irn = EdgeDisplacement(**kwargs)
        self.irn.load_state_dict(torch.load(pth), False)
        self.img_normal = TorchvisionNormalize()
        self.beta = 10
        self.exp_times = 8

    @torch.no_grad()
    def __call__(self, img: np.ndarray, cams: np.ndarray, threshold: Union[float, Tuple[float]], process_id=0) -> np.ndarray:
        with torch.cuda.device(process_id):
            self.irn.cuda().eval()
            img = imutils.HWC_to_CHW(self.img_normal(img))
            img = np.stack((img, np.flip(img, -1)))
            img, cams = torch.from_numpy(img).cuda(non_blocking=True), torch.from_numpy(cams).cuda(non_blocking=True)
            size = img.shape[-2:]
            edge, dp = self.irn(img)
            cams = F.interpolate(cams.unsqueeze(0), imutils.get_strided_size(size, 4), mode='bilinear', align_corners=False)[0]
            rw = indexing.propagate_to_edge(cams, edge, beta=self.beta, exp_times=self.exp_times, radius=5)
            rw_up = F.interpolate(rw, size, mode='bilinear', align_corners=False)[..., 0, :, :]
            rw_up /= rw_up.max()
            rw_up = rw_up.cpu().numpy()
            if isinstance(threshold, tuple):
                return tuple(np.pad(rw_up, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=th).argmax(0)
                            for th in threshold)
            return np.pad(rw_up, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold).argmax(0)
    
    def process_limit(self):
        return torch.cuda.device_count()

    
REFINER = {
    'none': NoneRefiner,
    'crf': DenseCRF,
    'irn': IRN,
}

def refiner_wrapper(refiner):
    if isinstance(refiner, str):
        return REFINER[refiner]()
    else:
        assert isinstance(refiner, REFINER)
        return refiner