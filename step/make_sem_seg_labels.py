import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, indexing, imutils
from misc.visualizeutils import save_palette_image

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            if 'cam' in cam_dict:
                cams = torch.tensor(cam_dict['cam'])
                cams = cams.unsqueeze(0) if cams.ndim == 2 else cams
            else:
                cams = torch.tensor(cam_dict['high_res'])
                cams = cams.unsqueeze(0) if cams.ndim == 2 else cams
                cams = F.interpolate(cams.unsqueeze(0), imutils.get_strided_size(orig_img_size, 4), mode='bilinear', align_corners=False)[0]
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            save_palette_image(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            # save cams
            cam_dict['cam'] = rw.cpu().numpy()
            cam_dict['high_res'] = rw_up.cpu().numpy()
            np.save(os.path.join(args.cam_out_dir + '_irn', img_name + '.npy'), cam_dict)

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), flush=True, end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
