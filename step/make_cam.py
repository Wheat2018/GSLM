import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
from misc.pyutils import print_func_params

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model, dataset, out_dir, num_workers):

    databin = dataset[process_id]
    data_loader = DataLoader(databin, shuffle=False, num_workers=num_workers, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)

            outputs = []
            for x in pack['img']:
                x = model(x[0].cuda(non_blocking=True))['cam']
                outputs.append(x[:1])
                outputs.append(x[1:].flip(-1))

            strided_cam = model.mixin_multi_cams(outputs, strided_size)[0]

            highres_cam = model.mixin_multi_cams(outputs, size)[0]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]

            highres_cam = highres_cam[valid_cat]

            # save cams
            np.save(os.path.join(out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == 0:
                print('%5.2f%%' % (iter / len(databin) * 100), flush=True, end='\b\b\b\b\b\b')

def make_cam(model, out_dir, img_list_path, voc12_root, scales=(1.0,), n_gpus=1, num_workers=1, show_params=True):
    if show_params:
        print_func_params(locals())

    model.eval()
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(img_list_path, voc12_root=voc12_root, scales=scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, out_dir, num_workers // n_gpus), join=True)
    print('100.00% ]')
    torch.cuda.empty_cache()

def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)

    make_cam(model, args.cam_out_dir, args.train_list, args.voc12_root, args.cam_scales, torch.cuda.device_count(), args.num_workers)