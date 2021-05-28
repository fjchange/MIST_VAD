import argparse
import os

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--MODEL',type=str,default='SHT_I3D',
                        help='the input should be in [UCF_C3D,UCF_I3D,SHT_C3D,SHT_I3D]')
    parser.add_argument('--expand_k',type=int,default=8)

    parser.add_argument('--batch_size',type=int,default=10)
    parser.add_argument('--dropout_rate',type=float,default=0.8)
    parser.add_argument('--segment_len',type=int,default=16)

    # for test time augmetation
    parser.add_argument('--test_ten_crop',dest='ten_crop',action='store_true')
    parser.set_defaults(ten_crop=False)

    parser.add_argument('--vis_UCF',dest='vis',action='store_true')
    parser.set_defaults(vis=False)

    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
    args.gpus=[i for i in range(len(args.gpus.split(',')))]

    if args.vis and args.MODEL!='UCF_C3D':
        args.vis=False

    return args