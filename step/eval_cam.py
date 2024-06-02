
from itertools import product
from torch import multiprocessing
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import torch
from misc import torchutils, imutils
from misc.pyutils import print_func_params
from net.refiner.refiner import DenseCRF, NoneRefiner
from voc12.dataloader import CAT_LIST, TorchvisionNormalize, cls_labels_dict
import torch.nn.functional as F

def from_npy(process_id, data, path):
    cam_dict = np.load(os.path.join(path, data['name'] + '.npy'), allow_pickle=True).item()
    cams = cam_dict['high_res']
    cams = cams[np.newaxis, ...] if cams.ndim == 2 else cams
    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
    return cams, keys

def from_seg_npy(process_id, data, path):
    cams = np.load(os.path.join(path, data['name'] + '.npy'))
    _, H, W = data['img'].shape
    cams = torch.FloatTensor(cams)[None, ...]
    cams = F.interpolate(cams, size=(H, W), mode="bilinear", align_corners=False)[0].numpy()
    cams[:, cams.argmax(0) == 0] = 0
    return cams[1:], np.arange(0, len(CAT_LIST) + 1)

def from_model(process_id, data, model, scales=(1.0,)):
    with torch.no_grad(), torch.cuda.device(process_id):
        model.cuda().eval()
        img = data['img']
        keys = np.nonzero(data['cls'])[0]

        cams = model.inference(img.cuda(non_blocking=True), scales)[0]
        cams = cams.cpu().numpy()[keys]
        keys = np.pad(keys + 1, (1, 0), mode='constant')

        return cams, keys

def _prepare(process_id, dataset, queue, threshold, img_normal, refiner, get_pred, kwargs: dict):
    databin = dataset[process_id]
    for idx in databin.indices:
        img, label = databin.dataset[idx]
        img = img.transpose((1, 2, 0)).astype(np.uint8)
        name = databin.dataset.ids[idx]
        cls = cls_labels_dict[int(name)]

        cams, keys = get_pred(process_id, {
            'name': name, 
            'img': torch.from_numpy(imutils.HWC_to_CHW(img_normal(img))), 
            'cls': cls, 
            'seg': label}, **kwargs)
        seg = refiner(img, cams, threshold, process_id=process_id)
        seg = keys[seg]
        queue.put((idx, seg))

def evaluate(get_pred, get_pred_kwargs: dict, 
             threshold=0.15, split='train', data_dir='auto', img_normal=TorchvisionNormalize(), refiner=NoneRefiner(),
             num_workers=1, show_params=True):
    num_workers = min(num_workers, refiner.process_limit())
    if show_params:
        print_func_params(locals())
    
    dataset = VOCSemanticSegmentationDataset(split=split, data_dir=data_dir)

    # get pred result
    split = torchutils.split_dataset(dataset, num_workers)
    queue = multiprocessing.Manager().Queue()

    print('[ ', end='')
    context = multiprocessing.spawn(_prepare, nprocs=num_workers, 
                                    args=(split, queue, threshold, img_normal, refiner, get_pred, get_pred_kwargs), 
                                    join=False)

    # calculate mIoU in parallel
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    idx, cnt = -1, 0
    def pred_it():
        nonlocal idx, cnt
        while cnt < len(dataset):
            try:
                pack = queue.get(timeout=0.1)
            except Exception:
                for q in context.error_queues:
                    if not q.empty():
                        raise Exception(q.get())
                continue
            idx = pack[0]
            print('%5.2f%%' % (cnt / len(dataset) * 100), flush=True, end='\b\b\b\b\b\b')
            cnt += 1
            yield pack[1]
    
    def gt_it():
        nonlocal idx, cnt
        while cnt < len(dataset):
            yield labels[idx]
    
    confusion = calc_semantic_segmentation_confusion(pred_it(), gt_it())
    context.join()
    print('100.00% ]', end=' ')
    torch.cuda.empty_cache()

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    res = {'iou': iou, 'miou': np.nanmean(iou)}
    print(res['miou'])
    return res

def run(args):
    print(evaluate(from_npy, {'path': args.cam_out_dir},
                   threshold=args.cam_eval_thres, 
                   split=args.chainer_eval_set, 
                   data_dir=args.voc12_root, 
                   num_workers=args.num_workers))