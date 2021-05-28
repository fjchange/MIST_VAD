from yacs.config import CfgNode

_C = CfgNode()

'''
The paths below should be modified to adapt your project

'''

############# 1. Model Paths #######################

## The pretrained model paths
_C.I3D_MODEL_PATH='pretrained/model_rgb.pth'
_C.C3D_MODEL_PATH='pretrained/C3D_Sport1M/pth'

## Trained Ckpts
_C.UCF_C3D_MODEL_PATH='ckpts/UCF_C3D_AUC_0.8143.pth'
_C.UCF_I3D_MODEL_PATH='ckpts/UCF_I3D_AUC_0.8230.pth'
_C.SHT_C3D_MODEL_PATH='ckpts/SHT_C3D_AUC_0.9313.pth'
_C.STH_I3D_MODEL_PATH='ckpts/SHT_I3D_AUC_0.9483.pth'

## Vis Paths
_C.VIS_DIR='outputs/'
_C.TEST_SPATIAL_ANNOTATION_PATH='data/Test_Spatial_Annotation.npy'
############ 2. UCF Data ###########################
_C.TRAIN_H5_PATH='data/UCFCrime-Frames.h5'
_C.TEST_H5_PATH='data/UCFCrime-Frames-test.h5'
_C.TESTING_TXT_PATH='data/Temporal_Anomaly_Annotation_New.txt'

############# 3. SHT Data ##########################
_C.SHT_TRAIN_H5_PATH='data/SHT_Frames.h5'
_C.SHT_TEST_H5_PATH='data/SHT_Frames.h5'

_C.SHT_TEST_MASK_DIR='data/test_frame_mask/'
_C.SHT_TRAIN_TXT_PATH='data/SH_Train_new.txt'
_C.SHT_TEST_TXT_PATH='data/SH_Test_NEW.txt'

_C.PSEUDO_LABEL_PATH_SHT_I3D='data/SHT_I3D_PLs.npy'
_C.PSEUDO_LABEL_PATH_SHT_C3D='data/SHT_C3D_PLs.npy'

############# 4. Dataset Related ###################
_C.DATASET=CfgNode()
_C.DATASET.MEAN=[0.45,0.45,0.45]
_C.DATASET.STD=[0.225,0.225,0.225]

_C.DATASET.CROP_SIZE=224
_C.DATASET.RESIZE=256

_C.DATASET.C3D_MEAN=[90.25,97.66,101.41]
_C.DATASET.C3D_STD=[1,1,1]


############# 5.training setting ##################
_C.SEED=0
_C.LOG_DIR='logs/'
_C.SUMMARY_DIR='summarys/'
_C.MODEL_DIR='train_ckpts/'

