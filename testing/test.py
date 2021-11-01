import os
import sys
sys.path.append('..')
import torch
from configs import constant,options
import models
from datasets.dataset import Test_Dataset_C3D,Test_Dataset_I3D,Test_Dataset_SHT_C3D,Test_Dataset_SHT_I3D
# from utils.balanced_dataparallel import BalancedDataParallel
from torch.utils.data import DataLoader
from utils.eval_utils import cal_auc,cal_score_gap,cal_false_alarm
from visualization.Grad_CAM import GradCAM
from visualization.CAM import visualize_CAM_with_clip
import random
import numpy as np
from tqdm import tqdm
import cv2
from apex import amp

_C=constant._C

def load_model(model,state_dict):
    new_dict={}
    for key,value in state_dict.items():
        new_dict[key[7:]]=value
    model.load_state_dict(new_dict)


def load_model_dataset(args):
    def worker_init(worked_id):
        np.random.seed(_C.SEED+worked_id)
        random.seed(_C.SEED+worked_id)
    model=getattr(models,args.MODEL.split('_')[1]+'_SGA_STD')(dropout_rate=args.dropout_rate,expand_k=args.expand_k,
                                  freeze_backbone=False,freeze_blocks=None).cuda().eval()
    opt_level = 'O1'
    amp.init(allow_banned=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
                                 )
    model,optimizer=amp.initialize(model,optimizer,opt_level=opt_level,keep_batchnorm_fp32=None)

    if args.MODEL.split('_')[-1]=='C3D':
        if args.MODEL=='UCF_C3D':
            dataset=Test_Dataset_C3D(_C.TEST_H5_PATH,_C.TESTING_TXT_PATH,args.segment_len,args.ten_crop)
            load_model(model,torch.load(_C.UCF_C3D_MODEL_PATH)['model'])
        else:
            dataset=Test_Dataset_SHT_C3D(_C.SHT_TEST_H5_PATH,_C.SHT_TEST_TXT_PATH,_C.SHT_TEST_MASK_DIR,
                                         args.segment_len,args.ten_crop)
            load_model(model,torch.load(_C.SHT_C3D_MODEL_PATH)['model'])
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=10,
                              worker_init_fn=worker_init,drop_last=False)
    else:
        if args.MODEL=='UCF_I3D':
            dataset=Test_Dataset_I3D(_C.TEST_H5_PATH,_C.TESTING_TXT_PATH,args.segment_len,args.ten_crop)
            load_model(model,torch.load(_C.UCF_I3D_MODEL_PATH)['model'])
        else:
            dataset=Test_Dataset_SHT_I3D(_C.SHT_TEST_H5_PATH,_C.SHT_TEST_TXT_PATH,_C.SHT_TEST_MASK_DIR,
                                         args.segment_len,args.ten_crop)
            load_model(model,torch.load(_C.SHT_I3D_MODEL_PATH)['model'])
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=5,
                              worker_init_fn=worker_init,drop_last=False)
    return model,dataloader

def eval_UCF(args,model,test_dataloader):
    total_labels, total_scores,normal_scores= [], [],[]

    if args.vis:
        gradcam=GradCAM(model.Regressor,grad_pp=False)
        test_spatial_annotation = np.load(_C.TEST_SPATIAL_ANNOTATION_PATH, allow_pickle=True).tolist()

    for frames,ano_types, keys, idxs,annos in tqdm(test_dataloader):
        frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()

        with torch.no_grad():
            scores, feat_maps = model(frames)[:2]

        if args.ten_crop:
            scores = scores.view([-1, 10, 2]).mean(dim=-2)
        for i, (clip, score, ano_type, key, idx, anno, feat_map) in enumerate(
                zip(frames, scores, ano_types, keys, idxs, annos,
                    feat_maps)):
            anno = anno.numpy().astype(int)
            if np.isnan(anno.any()):
                raise ValueError('NaN in anno')
            score = score.float().squeeze()[1].detach().cpu().item()
            if np.isnan(score):
                raise ValueError('NaN in score')
            anno = anno.astype(int)
            score = [score] * args.segment_len
            total_scores.extend(score)
            total_labels.extend(anno.tolist())
            if ano_type=='Normal':
                normal_scores.extend(score)
            if args.vis and ano_type != 'Normal':
                spa_annos = test_spatial_annotation[key]

                for f_idx in range(idx * args.segment_len, args.segment_len * (idx + 1)):
                    if f_idx in spa_annos.keys():
                        cam_map = gradcam(feat_map)
                        cam_path = _C.VIS_DIR+ '/{}-{}.jpg'.format(key, f_idx)
                        cam_clip = visualize_CAM_with_clip(cam_map, clip, (320, 240))
                        cv2.imwrite(cam_path, cam_clip)


    return eval(total_scores,total_labels,normal_scores)

def eval_SHT(model,test_dataloader):
    total_labels, total_scores,normal_scores = [], [],[]
    for frames,anno_type,_,annos in test_dataloader:
        frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()
        with torch.no_grad():
            scores, feat_maps = model(frames)[:2]
        if args.ten_crop:
            scores = scores.view([-1, 10, 2]).mean(dim=-2)
        for clip, score, anno in zip(frames, scores, annos):
            score = [score.squeeze()[1].detach().cpu().item()] * args.segment_len
            total_scores.extend(score)
            total_labels.extend(anno.tolist())
            if anno_type=='Normal':
                normal_scores.extend(score)
    return eval(total_scores,total_labels,normal_scores)

def eval(total_scores,total_labels,normal_scores):
    total_scores,total_labels=np.array(total_scores),np.array(total_labels)
    auc = cal_auc(total_scores, total_labels)
    far=cal_false_alarm(normal_scores,[0]*len(normal_scores))
    gap=cal_score_gap(total_scores,total_labels)
    print('{}: AUC {:.2f}%, FAR {:.2f}%, GAP {:.2f}%'.format(args.MODEL,auc*100,far*100,gap*100))

def test(args):
    model,dataloader=load_model_dataset(args)

    if args.MODEL.split('_')[0]=='UCF':
        eval_UCF(args,model,dataloader)
    else:
        eval_SHT(model,dataloader)

if __name__=='__main__':
    args=options.parse_args()
    test(args)



