import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader
from misc.pyutils import print_func_params
from misc.visualizeutils import Heatmap, voc12_colors, separator,read_palette_image
from net.refiner.refiner import NoneRefiner

import voc12.dataloader
from misc import torchutils

def _work(process_id, infer_dataset, cam_dir, pseudo_dir, out_dir, fg_thres, bg_thres, append=False):
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)
    
    colors =  Heatmap()
    for iter, pack in enumerate(infer_data_loader):
        img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()
        cam_dict = np.load(os.path.join(cam_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        cams = cams[np.newaxis, ...] if cams.ndim == 2 else cams
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        # 1. find confident fg & bg
        fg_conf_cam = NoneRefiner()(img, cams, fg_thres)
        fg_conf = keys[fg_conf_cam]

        bg_conf_cam = NoneRefiner()(img, cams, bg_thres)
        bg_conf = keys[bg_conf_cam]

        fg_conf[(fg_conf == 0) & (bg_conf != 0)] = 255
        seg = fg_conf

        conf = read_palette_image(os.path.join(pseudo_dir, img_name + '.png'))

        cam_color = (voc12_colors[seg.astype(np.int8)]).astype(np.uint8)
        crf_color = (voc12_colors[conf.astype(np.int8)]).astype(np.uint8)

        # color label
        out_img = np.hstack(separator((img, cam_color, crf_color)))
        name = os.path.join(out_dir, 'label', img_name + '.png')
        if append and os.path.exists(name):
            out_img = np.vstack((imageio.imread(name), out_img))
        imageio.imwrite(name, out_img)

        potential = []
        for cam in cams:
            stair = colors[cam]

            rate = 0.3
            pot = img * rate + stair * (1 - rate)

            potential.append(pot.astype(np.uint8))

        # potential
        out_img = np.hstack(separator(potential))
        name = os.path.join(out_dir, 'potential', img_name + '.png')
        if append and os.path.exists(name):
            out_img = np.vstack((imageio.imread(name), out_img))
        imageio.imwrite(name, out_img)

        if process_id == 0:
            print('%5.2f%%' % (iter / len(databin) * 100), flush=True, end='\b\b\b\b\b\b')

def visualize_pseudo(cam_dir, pseudo_dir, out_dir, fg_thres, bg_thres, img_list_path, voc12_root, 
                     append=False, num_workers=1, show_params=True):
    if show_params:
        print_func_params(locals())
    
    os.makedirs(os.path.join(out_dir, 'label'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'potential'), exist_ok=True)

    dataset = voc12.dataloader.VOC12ImageDataset(img_list_path, voc12_root=voc12_root, img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, num_workers)
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=num_workers, args=(dataset, cam_dir, pseudo_dir, out_dir, fg_thres, bg_thres, append), join=True)
    print('100.00% ]')

def run(args):
    visualize_pseudo(args.cam_out_dir, args.pseudo_out_dir, args.visualize_out_dir, args.conf_fg_thres, args.conf_bg_thres,
                     args.train_list, args.voc12_root, False, num_workers=args.num_workers)
