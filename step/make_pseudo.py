
import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader
from misc.pyutils import print_func_params
from net.refiner.refiner import NoneRefiner

import voc12.dataloader
from misc import torchutils, imutils
from misc.visualizeutils import save_palette_image

def _work(process_id, infer_dataset, cam_dir, out_dir, fg_thres, bg_thres, refiner):

    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()
        cam_dict = np.load(os.path.join(cam_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        cams = cams[np.newaxis, ...] if cams.ndim == 2 else cams
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        # 1. find confident fg & bg
        fg_pred, bg_pred = refiner(img, cams, (fg_thres, bg_thres), process_id=process_id)
        fg_conf = keys[fg_pred]
        bg_conf = keys[bg_pred]

        # 2. combine confident fg & bg
        fg_conf[(fg_conf == 0) & (bg_conf != 0)] = 255
        seg = fg_conf

        save_palette_image(os.path.join(out_dir, img_name + '.png'), seg.astype(np.uint8))

        if process_id == 0:
            print('%5.2f%%' % (iter / len(databin) * 100), flush=True, end='\b\b\b\b\b\b')

def make_pseudo(cam_dir, out_dir, fg_thres, bg_thres, img_list_path, voc12_root, refiner=NoneRefiner(), num_workers=1, show_params=True):
    num_workers = min(num_workers, refiner.process_limit())
    if show_params:
        print_func_params(locals())

    dataset = voc12.dataloader.VOC12ImageDataset(img_list_path, voc12_root=voc12_root, img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, num_workers)
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=num_workers, args=(dataset, cam_dir, out_dir, fg_thres, bg_thres, refiner), join=True)
    print('100.00% ]')

def run(args):
    make_pseudo(args.cam_out_dir, args.pseudo_out_dir, args.conf_fg_thres, args.conf_bg_thres,
                args.train_list, args.voc12_root, args.cam_refiner, args.num_workers)
