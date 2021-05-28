import torch
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import h5py


class SH_Train_Origin_Dataset(Dataset):
    def __init__(self, part_num, part_len, h5_path, train_txt, sample, pseudo_labels_path=None, norm=0):
        self.part_num = part_num
        self.part_len = part_len
        self.train_txt = train_txt
        self.norm = norm
        self.h5_path = h5_path
        self.sample = sample

        if pseudo_labels_path != None and os.path.exists(pseudo_labels_path):
            self.pseudo_labels = np.load(pseudo_labels_path, allow_pickle=True).tolist()
        else:
            self.pseudo_labels = None
        self.load_feat()
        self.shuffle_keys()

    def load_feat(self):
        self.norm_feats = []
        self.abnorm_feats = []
        lab_dict = {'Normal': 0, 'Abnormal': 1}
        h5 = h5py.File(self.h5_path, 'r')
        lines = open(self.train_txt, 'r').readlines()
        self.norm_keys = []
        self.abnorm_keys = []
        for line in lines:
            line_split = line.strip().split(',')
            label = int(line_split[-1])
            key = line_split[0]

            if label == 0:
                self.norm_feats.append(h5[key + '.npy'][:])
                if self.pseudo_labels != None:
                    self.norm_keys.append(key + '.npy')
            else:
                self.abnorm_feats.append(h5[key + '.npy'][:])
                if self.pseudo_labels != None:
                    self.abnorm_keys.append(key + '.npy')

    def __len__(self):
        return min(len(self.norm_feats), len(self.abnorm_feats))

    def shuffle_keys(self):
        self.norm_iters = np.random.permutation(len(self.norm_feats))
        self.abnorm_iters = np.random.permutation(len(self.abnorm_feats))

    def sample_feat(self, feat, labs, vid_type='Normal'):
        feat = np.array(feat)
        feat_len = feat.shape[0]

        if type(labs) == type(None):
            if vid_type == 'Normal':
                labs = np.ones([feat_len, 1], dtype=np.float32)
            else:
                labs = np.zeros([feat_len, 1], dtype=np.float32)

        else:
            # if feat_len!=labs.shape[0]:
            # print(feat_len,labs.shape[0])
            if labs.shape.__len__() == 2 and labs.shape[-1] == 2:
                labs = labs[:, -1]

        if self.sample == 'uniform':
            if (feat_len - self.part_len) // (self.part_num + 1) < 1:
                move = 0
            else:
                move = np.random.randint((feat_len - self.part_len) // (self.part_num + 1))
            # we want to have
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int) + move
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
        else:
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int)
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
            if chosen[1, 0] - chosen[0, 0] == 0:
                move = 0
            else:
                move = np.random.randint(0, chosen[1, 0] - chosen[0, 0], [self.part_num + 1]).repeat(
                    self.part_len).reshape([-1, self.part_len])
            chosen = chosen + move

        chosen = chosen.reshape([-1])

        return feat[chosen[:self.part_num * self.part_len], :], labs[chosen[:self.part_num * self.part_len]]

    def __getitem__(self, item):
        norm_iter = self.norm_iters[item]
        abnorm_iter = self.abnorm_iters[item]
        if self.pseudo_labels != None:
            norm_labs = self.pseudo_labels[self.norm_keys[norm_iter]]
            abnorm_labs = self.pseudo_labels[self.abnorm_keys[abnorm_iter]]
        else:
            norm_labs = abnorm_labs = None
        norm_feat, norm_labs = self.sample_feat(self.norm_feats[norm_iter], norm_labs, vid_type="Normal")
        abnorm_feat, abnorm_labs = self.sample_feat(self.abnorm_feats[abnorm_iter], abnorm_labs, vid_type='Abnormal')

        return torch.from_numpy(norm_feat).float(), torch.from_numpy(norm_labs).float(), \
               torch.from_numpy(abnorm_feat).float(), torch.from_numpy(abnorm_labs).float()

def shanghaitech_test(txt_path,mask_dir,h5_file,norm):
    lines=open(txt_path,'r').readlines()
    annos = []
    labels=[]
    names=[]
    h5=h5py.File(h5_file,'r')
    output_feats=[]
    for line in lines:
        line_split=line.strip().split(',')
        feat=h5[line_split[0].split('.')[0]+'.npy'][:]
        if norm==2:
            feat=feat/np.linalg.norm(feat,axis=-1,keepdims=True)
        if line_split[1]=='1':
            anno_npy_path=os.path.join(mask_dir,line_split[0]+'.npy')
            anno=np.load(anno_npy_path)
            # if anno.shape[0]%segment_len!=0:
            #     anno=np.sum(anno[:-(anno.shape[0]%segment_len)].reshape([-1,segment_len]),axis=-1,keepdims=False)
            # else:
            #     anno=np.sum(anno.reshape([-1,segment_len]),axis=-1,keepdims=False)
            # anno=anno.clip(0,1)
            labels.append('Abnormal')
        else:
            anno=np.zeros(int(line_split[-1]))
            labels.append('Normal')
        if anno.shape[0]==0:
            print(line)
        output_feats.append(feat)
        annos.append(anno)
        names.append(line_split[0])
    return output_feats,labels,annos#,names