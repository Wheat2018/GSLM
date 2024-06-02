import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils, imutils

def seg_criterion(input, target, label):
    res = []
    for inp, tg, cls in zip(input, target, label):
        mask = tg != 255
        if mask.sum() <= 0:
            continue
        inp = inp / (F.adaptive_max_pool2d(inp, (1, 1)) + 1e-5)
        inp = inp[:, mask] * 6.0
        tg = tg[mask]
        cls = torch.nonzero(cls)[:, 0]
        loss = sum([F.smooth_l1_loss(inp[c], (tg == c + 1) * 6.0 ) for c in cls])
        res.append(loss)
    return sum(res)
            

def run(args):
    train_dataset = voc12.dataloader.VOC12SegmentationDataset(args.train_list, voc12_root=args.voc12_root,
                                                              label_dir=args.pseudo_out_dir,
                                                              resize_long=(320, 640), hor_flip=True,
                                                              crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.fuse_cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.fuse_cam_batch_size) * args.fuse_cam_num_epoches

    for it in range(args.fuse_cam_num_iter):
        net = getattr(importlib.import_module(args.cam_network), 'Net')()
        if it <= 0:
            name = args.fuse_cam_weights_input
            net.load_state_dict(torch.load(name), strict=False)
        else:
            name = args.fuse_cam_weights_name.replace('.', f'_it{it}.')
            net.load_state_dict(torch.load(name), strict=True)
        print(f"load {name}")
        eval_thres = args.cam_eval_thres + args.fuse_cam_eval_thres_inc * it
        fg_thres = args.conf_fg_thres
        bg_thres = args.conf_bg_thres

        param_groups = net.trainable_parameters()
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.fuse_cam_learning_rate, 'weight_decay': args.fuse_cam_weight_decay},
            {'params': param_groups[1], 'lr': 10*args.fuse_cam_learning_rate, 'weight_decay': args.fuse_cam_weight_decay},
        ], lr=args.fuse_cam_learning_rate, weight_decay=args.fuse_cam_weight_decay, max_step=max_step)

        model = torch.nn.DataParallel(net).cuda()
        model.train()

        avg_meter = pyutils.AverageMeter()

        timer = pyutils.Timer()

        for ep in range(args.fuse_cam_num_epoches):

            print('Epoch %d/%d' % (ep+1, args.fuse_cam_num_epoches))

            for step, pack in enumerate(train_data_loader):

                img = pack['img']
                label = pack['cls'].cuda(non_blocking=True)
                seg = pack['label'].cuda(non_blocking=True)

                outputs = model(img.cuda())
                seg = F.interpolate(seg.unsqueeze(1), outputs['cam'].shape[-2:], mode='nearest')[:, 0, ...]
                cls_loss = F.multilabel_soft_margin_loss(outputs['cls'], label)
                seg_loss = seg_criterion(outputs['cam'], seg, label)
                
                loss = cls_loss + args.fuse_cam_seg_loss_rate * seg_loss
                record = {'cls': cls_loss.item(), 'seg': seg_loss.item(), 'all': loss.item()}
                avg_meter.add(record)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (optimizer.global_step-1)%100 == 0:
                    timer.update_progress(optimizer.global_step / max_step)

                    print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                        'loss:{cls: %.4f, seg: %.4f, all %.4f}' % (avg_meter.pop('cls'), avg_meter.pop('seg'), avg_meter.pop('all')),
                        'imps:%.1f' % ((step + 1) * args.fuse_cam_batch_size / timer.get_stage_elapsed()),
                        'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                        'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            name = args.fuse_cam_weights_name
            if it < args.fuse_cam_num_iter - 1:
                name = name.replace('.', f'_it{it + 1}.')
            torch.save(model.module.state_dict(), name)
            print(f'{name} saved.')
            torch.cuda.empty_cache()
            net.eval()

            print(f'it{it + 1} evaluate with refiner: ')
            from step.eval_cam import evaluate, from_model
            print(evaluate(from_model, {'model': net, 'scales': args.fuse_cam_scales, }, 
                           threshold=eval_thres, 
                           split=args.chainer_eval_set, 
                           data_dir=args.voc12_root,
                           refiner=args.fuse_cam_refiner,
                           num_workers=torch.cuda.device_count()))

            print(f'it{it + 1} make cam: ')
            from step.make_cam import make_cam
            make_cam(net, args.cam_out_dir, args.train_list, args.voc12_root,
                    args.fuse_cam_scales, torch.cuda.device_count(), args.num_workers)

            print(f'it{it + 1} make pseudo: ')
            from step.make_pseudo import make_pseudo
            make_pseudo(args.cam_out_dir, args.pseudo_out_dir, fg_thres, bg_thres,
                        args.train_list, args.voc12_root, args.fuse_cam_refiner, args.num_workers)
            
            if args.fuse_cam_visualize:
                print(f'it{it + 1} visualize pseudo: ')
                from step.visualize_pseudo import visualize_pseudo
                visualize_pseudo(args.cam_out_dir, args.pseudo_out_dir, args.visualize_out_dir, fg_thres, bg_thres,
                                 args.train_list, args.voc12_root, append=True, num_workers=args.num_workers)
            timer.reset_stage()
            torch.cuda.empty_cache()

