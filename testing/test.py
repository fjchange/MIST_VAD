import os
import sys
sys.path.extend(['..','../..','../../..'])

from configs import constant,options
import models
from datasets.dataset import Test_Dataset_C3D,Test_Dataset_I3D,Test_Dataset_SHT_C3D,Test_Dataset_SHT_I3D
# from utils.balanced_dataparallel import BalancedDataParallel
from torch.utils.data import DataLoader
from utils.eval_utils import cal_auc,cal_score_gap,cal_false_alarm
from visualization.Grad_CAM import GradCAM

_C=constant._C

def load_model_dataset(args):
    def worker_init(worked_id):
        np.random.seed(CFG.SEED+worked_id)
        random.seed(CFG.SEED+worked_id)
    model=getattr(models,args.MODEL.split('_')[0]+'_SGA_STD')(dropout_rate=args.dropout_rate,expand_k=args.expand_k,
                                  freeze_backbone=False,freeze_blocks=None).cuda().eval()
    if args.MODEL.split('_')[-1]=='C3D':
        if args.MODEL=='UCF_C3D':
            dataset=Test_Dataset_C3D(_C.TEST_H5_PATH,_C.TESTING_TXT_PATH,args.segment_len,args.ten_crop)
            model.load_state_dict(torch.load(_C.UCF_C3D_MODEL_PATH)['state_dict'])
        else:
            dataset=Test_Dataset_SHT_C3D(_C.SHT_TEST_H5_PATH,_C.SHT_TEST_TXT_PATH,_C.SHT_TEST_MASK_DIR,
                                         args.segment_len,args.ten_crop)
            model.load_state_dict(torch.load(_C.SHT_C3D_MODEL_PATH)['state_dict'])
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=10,
                              worker_init_fn=worker_init,drop_last=False)
    else:
        if args.MODEL=='UCF_I3D':
            dataset=Test_Dataset_I3D(_C.TEST_H5_PATH,_C.TESTING_TXT_PATH,args.segment_len,args.ten_crop)
            model.load_state_dict(torch.load(_C.UCF_I3D_MODEL_PATH)['state_dict'])
        else:
            dataset=Test_Dataset_SHT_I3D(_C.SHT_TEST_H5_PATH,_C.SHT_TEST_TXT_PATH,_C.SHT_TEST_MASK_DIR,
                                         args.segment_len,args.ten_crop)
            model.load_state_dict(torch.load(_C.SHT_I3D_MODEL_PATH)['state_dict'])
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=5,
                              worker_init_fn=worker_init,drop_last=False)
    return model,dataloader

def eval_UCF(args,model,test_dataloader):
    total_labels, total_scores= [], []

    data_iter=test_dataloader.__iter__()
    next_batch=data_iter.__next__()
    next_batch[0]=next_batch[0].cuda(non_blocking=True)
    if args.vis:
        gradcam=Grad_CAM(model.Regressor,grad_pp=False)
        test_spatial_annotation = np.load(_C.TEST_SPATIAL_ANNOTATION_PATH, allow_pickle=True).tolist()

    for frames,_,_,_,annos in tqdm(test_dataloader):
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

            if args.vis and ano_type != 'Normal':
                anomaly_scores.extend(score)
                anomaly_labels.extend(anno.tolist())

                spa_annos = test_spatial_annotation[key]

                for f_idx in range(idx * args.segment_len, args.segment_len * (idx + 1)):
                    if f_idx in spa_annos.keys():
                        cam_map = gradcam(feat_map)
                        cam_path = _C.VIS_DIR+ '/{}-{}.jpg'.format(key, f_idx)
                        cam_clip = visualize_CAM_with_clip(cam_map, clip, (320, 240))
                        cv2.imwrite(cam_path, cam_clip)

        return eval(total_scores, total_labels )

    return eval(total_scores,total_labels)

def eval_SHT(model,test_dataloader):
    total_labels, total_scores = [], []
    for frames,_,_,_,annos in test_dataloader:
        frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()
        with torch.no_grad():
            scores, feat_maps = model(frames)[:2]
        if args.ten_crop:
            scores = scores.view([-1, 10, 2]).mean(dim=-2)
        for clip, score, anno in zip(frames, scores, annos):
            score = [score.squeeze()[1].detach().cpu().item()] * args.segment_len
            total_scores.extend(score)
            score_dict[ano_type].extend(score)
            total_labels.extend(anno.tolist())
            label_dict[ano_type].extend(anno.tolist())

    return eval(total_scores,total_labels)

def eval(total_scores,total_labels):
    total_scores,total_labels=np.array(total_scores),np.array(total_labels)
    auc = cal_auc(total_scores, total_labels)
    far=cal_false_alarm(total_scores,total_labels)
    gap=cal_false_alarm(total_scores,total_labels)
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