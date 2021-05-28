import torch
from torch.utils.data import Dataset
from opencv_videovision import transforms
import numpy as np
import h5py
import cv2
import os

def read_testing_txt(file_path):
    vids_dict={}
    with open(file_path) as f:
        lines=f.readlines()
        for line in lines:
            line_list=line.strip().split('\t')
            annotations=[]
            for i in range(2,6):
                if int(line_list[i])==-1:
                    break
                else:
                    annotations.append(int(line_list[i]))

            vids_dict[line_list[0]]=[line_list[1],annotations,int(line_list[-1])]

    return vids_dict

Abnormal_type=['Abuse','Arrest','Arson','Assault','Burglary',
               'Explosion','Fighting','RoadAccidents','Robbery',
               'Shooting','Shoplifting','Stealing','Vandalism','Normal']

class Test_Dataset_I3D(Dataset):
    def __init__(self, h5_file, test_txt, segment_len, ten_crop=False, height=256, width=340, crop_size=224):
        self.h5_path = h5_file
        self.keys = sorted(list(h5py.File(self.h5_path, 'r').keys()))
        self.test_dict = read_testing_txt(test_txt)
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.crop_size = crop_size

        self.height = height
        self.width = width

        self.test_dict_annotation()

        self.mean = [128,128,128]
        self.std = [128, 128, 128]
        if ten_crop:
            self.ten_crop_aug = transforms.Compose([transforms.Resize([self.height, self.width]),
                                                    transforms.ClipToTensor(div_255=False),
                                                    transforms.Normalize(mean=self.mean, std=self.std),
                                                    transforms.TenCropTensor(self.crop_size)])

        self.transforms = transforms.Compose([transforms.Resize([self.height, self.width]),
                                              # transforms.CenterCrop(self.crop_size),
                                              transforms.ClipToTensor(div_255=False),
                                              transforms.Normalize(mean=self.mean, std=self.std)])
        self.dataset_len = len(h5py.File(self.h5_path, 'r')[self.keys[0]][:])
        # self.readin_h5()

    def __len__(self):
        return len(self.keys)

    def test_dict_annotation(self):
        self.annotation_dict = {}
        for key in self.test_dict.keys():
            ano_type, anno, frames_num = self.test_dict[key]
            annotation = np.zeros(frames_num - frames_num % (self.segment_len), dtype=int)
            if len(anno) >= 2:
                front = anno[0]
                back = anno[1]
                if front < annotation.shape[0]:
                    annotation[front:min(back, annotation.shape[0])] = 1
            if len(anno) == 4:
                front = anno[-2]
                back = anno[-1]
                if front < annotation.shape[0]:
                    annotation[front:min(back, annotation.shape[0])] = 1
            self.annotation_dict[key] = annotation

        key_dict = {}
        for key in self.keys:
            if key.split('-')[0] in self.annotation_dict.keys():
                self.keys.append(key)
                if key.split('-')[0] in key_dict.keys():
                    key_dict[key.split('-')[0]] += 1
                else:
                    key_dict[key.split('-')[0]] = 1

        for key in key_dict.keys():
            try:
                assert self.annotation_dict[key].shape[0] // self.segment_len == key_dict[key]
            except AssertionError:
                print(key, self.annotation_dict[key].shape[0] // self.segment_len, key_dict[key])

    def decode_imgs(self, frames):
        new_frames = []  # np.empty([self.segment_len, 240, 320, 3], dtype=np.uint8)
        # choices=np.linspace(0,self.dataset_len,self.segment_len+1,dtype=int)[:-1]
        # for i,choice in enumerate(choices):
        for i, frame in enumerate(frames):
            new_frames.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))

        # new_frames=torch.from_numpy(new_frames).float().permute([3,0,1,2])
        new_frames = self.transforms(new_frames)
        return new_frames

    def __getitem__(self, i):
        key = self.keys[i]
        # frames=h5py.File(self.h5_path,'r')[key][:]
        with h5py.File(self.h5_path, 'r') as h5:
            frames = h5[key][:]
        key_tmp, idx = key.split('-')
        idx = int(idx)
        anno = self.annotation_dict[key_tmp + '.mp4'][
               idx * self.segment_len:(idx + 1) * self.segment_len]
        if 'Normal' in key_tmp:
            ano_type = 'Normal'
        else:
            ano_type = key_tmp.split('_')[0][:-3]
        # begin=time.time()
        frames = self.decode_imgs(frames)
        # end=time.time()
        # print('take time {}'.format(end-begin))
        return frames, ano_type, key_tmp, idx, anno

class Test_Dataset_C3D(Dataset):
    def __init__(self,h5_file,test_txt,segment_len,ten_crop=False,height=128,width=171,crop_size=112):
        self.h5_path = h5_file
        self.keys = sorted(list(h5py.File(self.h5_path, 'r').keys()))
        self.test_dict = read_testing_txt(test_txt)
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.crop_size = crop_size

        self.height=height
        self.width=width

        self.test_dict_annotation()


        self.mean=[90.25,97.66,101.41]
        self.std=[1,1,1]
        if ten_crop:
            # self.resize_aug=transforms.Resize([256,340])
            # self.ten_crop_aug=transforms.TenCropTensor(self.crop_size)
            self.ten_crop_aug = transforms.Compose([transforms.Resize([self.height, self.width]),
                                                    transforms.ClipToTensor(div_255=False),
                                                    transforms.Normalize(mean=self.mean,std=self.std),
                                                    transforms.TenCropTensor(self.crop_size)])


        self.transforms = transforms.Compose([ transforms.Resize([self.height, self.width]),
                                               transforms.CenterCrop(self.crop_size),
                                                transforms.ClipToTensor(div_255=False),
                                                transforms.Normalize(mean=self.mean, std=self.std)])
        self.dataset_len = len(h5py.File(self.h5_path,'r')[self.keys[0]][:])
        # self.readin_h5()

    def __len__(self):
        return len(self.keys)

    # def readin_h5(self):
    #     self.data_dict = {}
    #     with h5py.File(self.h5_path, 'r') as h5:
    #         for key in h5.keys():
    #             self.data_dict[key] = h5[key][:]

    def test_dict_annotation(self):
        self.annotation_dict = {}
        for key in self.test_dict.keys():
            ano_type, anno, frames_num = self.test_dict[key]
            annotation = np.zeros(frames_num - frames_num % (self.segment_len), dtype=int)
            if len(anno) >= 2:
                front = anno[0]
                back = anno[1]
                if front < annotation.shape[0]:
                    annotation[front:min(back, annotation.shape[0])] = 1
            if len(anno) == 4:
                front = anno[-2]
                back = anno[-1]
                if front < annotation.shape[0]:
                    annotation[front:min(back, annotation.shape[0])] = 1
            self.annotation_dict[key] = annotation

        key_dict={}
        for key in self.keys:
            if key.split('-')[0] in self.annotation_dict.keys():
                self.keys.append(key)
                if key.split('-')[0] in key_dict.keys():
                    key_dict[key.split('-')[0]]+=1
                else:
                    key_dict[key.split('-')[0]] =1
        # import pdb
        #         # pdb.set_trace()
        for key in key_dict.keys():
            try:
                assert self.annotation_dict[key].shape[0]//self.segment_len==key_dict[key]
            except AssertionError:
                print(key,self.annotation_dict[key].shape[0]//self.segment_len,key_dict[key])

    def decode_imgs(self, frames):
        new_frames = []#np.empty([self.segment_len, 240, 320, 3], dtype=np.uint8)
        # choices=np.linspace(0,self.dataset_len,self.segment_len+1,dtype=int)[:-1]
        # for i,choice in enumerate(choices):
        for i, frame in enumerate(frames):
            new_frames.append(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR))

        # new_frames=torch.from_numpy(new_frames).float().permute([3,0,1,2])
        if not self.ten_crop:
            new_frames = self.transforms(new_frames)
        else:
            new_frames=self.ten_crop_aug(new_frames)
            new_frames=torch.stack(new_frames,dim=0)
        return new_frames

    def __getitem__(self, i):
        key = self.keys[i]
        # frames=h5py.File(self.h5_path,'r')[key][:]
        with h5py.File(self.h5_path,'r') as h5:
            frames = h5[key][:]
        key_tmp, idx = key.split('-')
        idx = int(idx)
        anno = self.annotation_dict[key_tmp + '.mp4'][
               idx * self.segment_len :(idx + 1) * self.segment_len ]
        if 'Normal' in key_tmp:
            ano_type = 'Normal'
        else:
            ano_type = key_tmp.split('_')[0][:-3]
        # begin=time.time()

        frames = self.decode_imgs(frames)
        # end=time.time()
        # print('take time {}'.format(end-begin))
        return frames, ano_type, key_tmp, idx, anno

class Test_Dataset_SHT_C3D(Dataset):
    def __init__(self,h5_file,test_txt,test_mask_dir,segment_len=16,ten_crop=False,height=128,width=171,crop_size=112):
        self.h5_path = h5_file
        # self.keys = sorted(list(h5py.File(self.h5_path, 'r').keys()))
        self.test_txt=test_txt
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.crop_size = crop_size
        self.test_mask_dir=test_mask_dir

        self.mean=[90.25,97.66,101.41]
        self.std=[1,1,1]
        self.height=height
        self.width=width

        self.test_dict_annotation()
        if ten_crop:
            self.ten_crop_aug = transforms.Compose([transforms.Resize([self.height, self.width]),
                                                    transforms.ClipToTensor(div_255=False),
                                                    transforms.Normalize(mean=self.mean,std=self.std),
                                                    transforms.TenCropTensor(self.crop_size)])


        self.transforms = transforms.Compose([ transforms.Resize([self.height, self.width]),
                                               transforms.CenterCrop(self.crop_size),
                                                transforms.ClipToTensor(div_255=False),
                                                transforms.Normalize(mean=self.mean, std=self.std)])
        self.dataset_len = len(h5py.File(self.h5_path,'r')[self.keys[0]][:])

    def __len__(self):
        return len(self.keys)

    def test_dict_annotation(self):
        self.annotation_dict = {}
        self.keys=[]
        keys=sorted(list(h5py.File(self.h5_path, 'r').keys()))
        for line in open(self.test_txt,'r').readlines():
            key,anno_type,frames_num = line.strip().split(',')
            frames_num=int(frames_num)
            if anno_type=='1':
                label='Abnormal'
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))#[
                       #:frames_num - frames_num % self.segment_len]
            else:
                label='Normal'
                anno=np.zeros(frames_num-frames_num % self.segment_len,dtype=np.uint8)
            self.annotation_dict[key]=[anno,label]
        # key_dict={}
        for key in keys:
            if key.split('-')[0] in self.annotation_dict.keys():
                self.keys.append(key)
        #         if key.split('-')[0] in key_dict.keys():
        #             key_dict[key.split('-')[0]]+=1
        #         else:
        #             key_dict[key.split('-')[0]] =1
        # # import pdb
        # #         # pdb.set_trace()
        # for key in key_dict.keys():
        #     try:
        #         assert self.annotation_dict[key][0].shape[0]//self.segment_len==key_dict[key]
        #     except AssertionError:
        #         print(key,self.annotation_dict[key][0].shape[0]//self.segment_len,key_dict[key])

    def decode_imgs(self, frames):
        new_frames = []
        for i, frame in enumerate(frames):
            new_frames.append(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR))

        # new_frames=torch.from_numpy(new_frames).float().permute([3,0,1,2])
        new_frames = self.transforms(new_frames)
        return new_frames

    def __getitem__(self, i):
        key = self.keys[i]
        # frames=h5py.File(self.h5_path,'r')[key][:]
        with h5py.File(self.h5_path,'r') as h5:
            frames = h5[key][:]
        key_tmp, idx = key.split('-')
        idx = int(idx)
        ano_type=self.annotation_dict[key_tmp][1]
        if ano_type=='Normal':
            anno=np.zeros([self.segment_len],dtype=np.uint8)
        else:
            anno = self.annotation_dict[key_tmp][0][
                   idx * self.segment_len :(idx + 1) * self.segment_len ].astype(np.uint8)
        # begin=time.time()
        frames = self.decode_imgs(frames)
        # end=time.time()
        # print('take time {}'.format(end-begin))
        return frames, ano_type, idx, anno

class Test_Dataset_SHT_I3D(Dataset):
    def __init__(self,h5_file,test_txt,test_mask_dir,segment_len=16,ten_crop=False,height=256,width=340,crop_size=224):
        self.h5_path = h5_file
        # self.keys = sorted(list(h5py.File(self.h5_path, 'r').keys()))
        self.test_txt=test_txt
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.crop_size = crop_size
        self.test_mask_dir=test_mask_dir

        self.mean=[128,128,128]
        self.std=[128,128,128]
        self.height=height
        self.width=width

        self.test_dict_annotation()
        if ten_crop:
            self.ten_crop_aug = transforms.Compose([transforms.Resize([self.height, self.width]),
                                                    transforms.ClipToTensor(div_255=False),
                                                    transforms.Normalize(mean=self.mean,std=self.std),
                                                    transforms.TenCropTensor(self.crop_size)])


        self.transforms = transforms.Compose([ transforms.Resize([240, 320]),
                                               # transforms.CenterCrop(self.crop_size),
                                                transforms.ClipToTensor(div_255=False),
                                                transforms.Normalize(mean=self.mean, std=self.std)])
        self.dataset_len = len(h5py.File(self.h5_path,'r')[self.keys[0]][:])

    def __len__(self):
        return len(self.keys)

    def test_dict_annotation(self):
        self.annotation_dict = {}
        self.keys=[]
        keys=sorted(list(h5py.File(self.h5_path, 'r').keys()))
        for line in open(self.test_txt,'r').readlines():
            key,anno_type,frames_num = line.strip().split(',')
            frames_num=int(frames_num)
            if anno_type=='1':
                label='Abnormal'
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))#[
                       #:frames_num - frames_num % self.segment_len]
            else:
                label='Normal'
                anno=np.zeros(frames_num-frames_num % self.segment_len,dtype=np.uint8)
            self.annotation_dict[key]=[anno,label]
        # key_dict={}
        for key in keys:
            if key.split('-')[0] in self.annotation_dict.keys():
                self.keys.append(key)
        #         if key.split('-')[0] in key_dict.keys():
        #             key_dict[key.split('-')[0]]+=1
        #         else:
        #             key_dict[key.split('-')[0]] =1
        # # import pdb
        # #         # pdb.set_trace()
        # for key in key_dict.keys():
        #     try:
        #         assert self.annotation_dict[key][0].shape[0]//self.segment_len==key_dict[key]
        #     except AssertionError:
        #         print(key,self.annotation_dict[key][0].shape[0]//self.segment_len,key_dict[key])

    def decode_imgs(self, frames):
        new_frames = []
        for i, frame in enumerate(frames):
            new_frames.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))

        # new_frames=torch.from_numpy(new_frames).float().permute([3,0,1,2])
        new_frames = self.transforms(new_frames)
        return new_frames

    def __getitem__(self, i):
        key = self.keys[i]
        # frames=h5py.File(self.h5_path,'r')[key][:]
        with h5py.File(self.h5_path,'r') as h5:
            frames = h5[key][:]
        key_tmp, idx = key.split('-')
        idx = int(idx)
        ano_type=self.annotation_dict[key_tmp][1]
        if ano_type=='Normal':
            anno=np.zeros([self.segment_len],dtype=np.uint8)
        else:
            anno = self.annotation_dict[key_tmp][0][
                   idx * self.segment_len :(idx + 1) * self.segment_len ].astype(np.uint8)
        # begin=time.time()
        frames = self.decode_imgs(frames)
        # end=time.time()
        # print('take time {}'.format(end-begin))
        return frames, ano_type, idx, anno
