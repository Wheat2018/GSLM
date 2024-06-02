import argparse
import os

from misc import pyutils
from net.refiner.refiner import REFINER, refiner_wrapper

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", default="/mnt/data/VOC2012", type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.extractor.resnet50_cam", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    parser.add_argument("--cam_refiner", default='crf', type=refiner_wrapper, choices=REFINER.keys())
    parser.add_argument("--fuse_cam_refiner", default='crf', type=refiner_wrapper, choices=REFINER.keys())

    # Fuse Class Activation Map
    parser.add_argument("--fuse_cam_network", default="net.extractor.resnet50_wam", type=str)
    parser.add_argument("--fuse_cam_weights_input", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--fuse_cam_weights_name", default="sess/res50_wam.pth", type=str)
    parser.add_argument("--fuse_cam_seg_loss_rate", default=0.5, type=float)
    parser.add_argument("--fuse_cam_batch_size", default=16, type=int)
    parser.add_argument("--fuse_cam_num_epoches", default=10, type=int)
    parser.add_argument("--fuse_cam_num_iter", default=3, type=int)
    parser.add_argument("--fuse_cam_learning_rate", default=0.001, type=float)
    parser.add_argument("--fuse_cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--fuse_cam_eval_thres_inc", default=0.1, type=float)
    parser.add_argument("--fuse_cam_visualize", default=True,
                        help="whether visualiaze the pseudo label. It will take over 30 minutes for each iteration.")
    parser.add_argument("--fuse_cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    parser.add_argument("--eval_cam_range", default=(0.10, 0.90, 0.01),
                        help="range of cam evaluation")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.refiner.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=16, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--pseudo_out_dir", default="result/pseudo", type=str)
    parser.add_argument("--visualize_out_dir", default="result/visualize", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--make_pseudo_pass", default=True)
    parser.add_argument("--visualize_pseudo_pass", default=False)
    parser.add_argument("--train_fuse_cam_pass", default=True)
    parser.add_argument("--train_irn_pass", default=False)
    parser.add_argument("--make_sem_seg_pass", default=False)
    parser.add_argument("--eval_sem_seg_pass", default=False)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.pseudo_out_dir, exist_ok=True)
    os.makedirs(args.visualize_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)
    

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.make_pseudo_pass is True:
        import step.make_pseudo

        timer = pyutils.Timer('step.make_pseudo:')
        step.make_pseudo.run(args)

    if args.visualize_pseudo_pass is True:
        import step.visualize_pseudo

        timer = pyutils.Timer('step.visualize_pseudo:')
        step.visualize_pseudo.run(args)

    if args.train_fuse_cam_pass is True:
        import step.train_fuse_cam

        timer = pyutils.Timer('step.train_fuse_cam:')
        step.train_fuse_cam.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

