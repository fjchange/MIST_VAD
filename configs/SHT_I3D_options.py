import argparse
import os
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--MODEL', type=str, default='I3D')
    parser.add_argument('--train', type=str, default='SGA')
    parser.add_argument('--expand_k',type=int,default=8)
    parser.add_argument('--label_smoothing',type=float,default=0)
    parser.add_argument('--hard_label',dest='use_hard_label',action='store_true')
    parser.set_defaults(use_hard_label=False)
    parser.add_argument('--continuous_sampling',dest='use_continuous_sampling',action='store_true')
    parser.set_defaults(use_continuous_sampling=False)

    parser.add_argument('--batch_size',type=int,default=10)
    parser.add_argument('--clip_num',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=301)
    parser.add_argument('--accumulate_step',type=int,default=3)

    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--min_lr',type=float,default=1e-6)
    parser.add_argument('--max_step',type=int,default=20)
    parser.add_argument('--optim',type=str,default='adam')
    parser.add_argument('--dropout_rate',type=float,default=0.8)
    parser.add_argument('--grad_clip',type=float,default=20)
    parser.add_argument('--warmup_epochs',type=int,default=5)

    parser.add_argument('--use_bn_sta',dest='freeze_bn_sta',action='store_false')
    parser.set_defaults(freeze_bn_sta=True)
    parser.add_argument('--freeze_bn',dest='train_bn',action='store_false')
    parser.set_defaults(train_bn=True)

    parser.add_argument('--freeze_backbone',dest='train_backbone',action='store_false')
    parser.set_defaults(train_backbone=True)
    parser.add_argument('--freeze_blocks',type=str,default='conv3d_1a_7x7,conv3d_2b_1x1,conv3d_2c_3x3,mixed_3b,mixed_3c,mixed_4b,mixed_4c,mixed_4d,mixed_4e,mixed_4f,mixed_5b,mixed_5c')
    parser.add_argument('--pretrained_path',type=str)

    parser.add_argument('--train_all',dest='pretrained_backbone',action='store_false')
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument('--freeze_epochs',type=int,default=30)
    parser.add_argument('--freeze_bn_epochs',type=int,default=30)

    parser.add_argument('--segment_len',type=int,default=16)

    parser.add_argument('--gpus',type=str,default='0,1,2')
    parser.add_argument('--gpu0sz',type=float,default=0.8)

    # for test time augmetation
    parser.add_argument('--test_ten_crop',dest='ten_crop',action='store_true')
    parser.set_defaults(ten_crop=False)
    # loss balance hypermeters
    parser.add_argument('--lambda_atten',type=float,default=1.0)
    parser.add_argument('--lambda_base',type=float,default=1.0)

    parser.add_argument('--class_reweights',type=str,default='0.8,0.65')

    # threshold for iou calculate
    parser.add_argument('--gradcam_pp',dest='grad_pp',action='store_true')
    parser.set_defaults(grad_pp=False)

    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
    args.gpus=[i for i in range(len(args.gpus.split(',')))]
    args.freeze_blocks=[i for i in args.freeze_blocks.split(',')]
    args.class_reweights=[float(i) for i in args.class_reweights.split(',')]

    return args
