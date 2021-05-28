import torch
from torch import nn
import numpy as np
from torch.autograd import Function
import cv2
import torch.nn.functional as F

class Simple_Regressor(nn.Module):
    def __init__(self,input_feature_dim,dropout_rate=0.6):
        super(Simple_Regressor, self).__init__()

        self.regressor=nn.Sequential(nn.Linear(input_feature_dim,512),nn.ReLU(),nn.Dropout(dropout_rate),
                                     nn.Linear(512,32),nn.Dropout(dropout_rate),
                                     nn.Linear(32,2),nn.Sigmoid())
        self.GAP=nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # x=self.GAP(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x=x.view([-1,x.shape[-1]])
        logits=self.regressor(x)
        return logits
grads=[]
def hook(grad):
    grads.append(grad)

def test_GradCatcher():
    model=Simple_Regressor(1024)
    old_state_dict=torch.load('/data2/jiachang/Weakly_Supervised_VAD/models/single_I3D_RGB_550')
    model.load_state_dict({'r'+k[1:]:v for k,v in old_state_dict.items()})
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    target_layer_name='0'
    # grad_catcher=GradCatcher(model,target_layer_name)
    input=torch.from_numpy(np.random.random([1,1024])).float().requires_grad_()
    h=input.register_hook(hook)
    # a,x=grad_catcher(input)
    # grad_catcher.zero_grad()
    output=model(input)
    model.zero_grad()
    one_hot=torch.zeros([1,2]).float()
    one_hot[0][-1]=1
    one_hot=torch.sum(one_hot*output)
    one_hot.backward()

    print(grads[0].max(),grads[0].min())
    # print(grad_catcher.gradients)
    # print(a)

class GradCAM:
    def __init__(self,model,grad_pp=False,index=1,att=False,expand_k=8,act=True):
        self.model=model
        self.gradients=[]
        self.grad_pp=grad_pp
        self.index=index
        self.att=att
        self.expand_k=expand_k
        self.act=act
    def hook_tensor_gradients(self,grad):
        self.gradients.append(grad)

    def __call__(self,feature_maps):
        # supposed that input is with shape [C,T,H,W]
        # self.model.eval()
        # for name,p in self.model.named_parameters():
        #     p.requires_grad_(False)
        self.model=self.model.eval()
        self.gradients=[]
        if not self.att:
            # import pdb
            # pdb.set_trace()
            if feature_maps.shape.__len__()==5:
                feature_maps=feature_maps
            elif feature_maps.shape.__len__()==4:
                # [C,T,H,W]->[1,C,H,W]
                feature_maps = feature_maps.unsqueeze(0)
            elif feature_maps.shape.__len__()==3:
                feature_maps=feature_maps.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError('feature_maps should be[N,C,T,H,W], [C,T,H,W] or [C,H,W], but get shape as {}'.format(feature_maps.shape))
            # import pdb
            # pdb.set_trace()
            feat_map=feature_maps.clone().requires_grad_(True)
            h=feat_map.register_hook(self.hook_tensor_gradients)
            feat=feat_map.mean(dim=-1).mean(dim=-1).mean(dim=-1)
            if self.act==False:
                logits = torch.softmax(self.model(feat),dim=-1)
            else:
                logits=self.model(feat)
        else:
            # import pdb
            # pdb.set_trace()
            # [N,C,T,H,W] or [C,T,H,W]
            if feature_maps.shape.__len__()==4:
                feature_maps=feature_maps.unsqueeze(0)
            feat_map=feature_maps.clone().requires_grad_(True)
            h=feat_map.register_hook(self.hook_tensor_gradients)

            # [N,C]->[C]
            logits=self.model(feat_map).mean(dim=-1).mean(dim=-1).mean(dim=-1)
            logits=torch.softmax(logits.view(-1,2,self.expand_k).mean(dim=-1),dim=-1)
        # import pdb
        # pdb.set_trace()
        # cams=np.zeros([2,feat_map.shape[-2],feature_maps.shape[-1]],dtype=np.float32)
        # for e in range(2):
        # self.gradients=[]
        # if e==0:
        #     logits[0][e].backward(retain_graph=True)
        # else:
        logits[0][1].backward()
        # logits with [1,16,1,1,1]
        if not self.grad_pp:
            # one_hot.backward()
            # h.remove()
            # print(self.gradients[0].shape,feature_maps[0].shape)
            # import pdb
            # pdb.set_trace()
            weights=self.gradients[0].cpu().numpy().mean(axis=(2,3,4)).squeeze()

            target=feature_maps[0].cpu().numpy()

            cam=np.zeros(target.shape[1:])
            # print(weights)
            for i,w in enumerate(weights):
                cam+=w*target[i,]
            cam = np.maximum(cam, 0)
            cam=cam.mean(axis=0)
            cam = cv2.resize(cam, (feature_maps.shape[-1],feature_maps.shape[-2]))
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam)+1e-8)
            # cam = logits[0][1].cpu().item() * cam
            return cam
            # cams[e]=cam
        # import pdb
        # pdb.set_trace()
        else:

            # print(self.gradients[0].shape,feature_maps[0].shape)
            # [B,C,H,W]
            gradients = self.gradients[0].cpu()
            activations = feature_maps.cpu()
            b, k, u, v = gradients.size()

            alpha_num = gradients.pow(2)
            alpha_denom = gradients.pow(2).mul(2) + \
                          activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

            alpha = alpha_num.div(alpha_denom + 1e-7)
            positive_gradients = F.relu(logits[0][1].exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

            saliency_map = (weights*activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map).detach().cpu().numpy().squeeze(0)
            # if self.att:
            saliency_map=cv2.resize(saliency_map[0].mean(dim=2),(feat_map.shape[-1],feat_map.shape[-2]))
            # saliency_map=cv2.resize(saliency_map[0],(feat_map.shape[-1],feat_map.shape[-2]))
            # [1,H,W]
            saliency_map=(saliency_map-saliency_map.min())/(saliency_map.max()-saliency_map.min())
            # cams[i]=saliency_map

            return saliency_map


import sys
sys.path.append("..")
sys.path.append("../..")
from UCFCrime.SpatioTemporal_Detection.CAM import visualize_CAM
def test_gradcam():
    model=Simple_Regressor(1024).cuda().eval()
    old_state_dict=torch.load('/data2/jiachang/Weakly_Supervised_VAD/models/single_I3D_RGB_550')
    model.load_state_dict({'r'+k[1:]:v for k,v in old_state_dict.items()})
    feature_maps=torch.from_numpy(np.random.random([1024,2,8,10])).float().cuda()
    gradcam=GradCAM(model,grad_pp=False)
    cam=gradcam(feature_maps)
    print(cam.shape)
    visualize_CAM(cam,(320,240))

import sys
sys.path.append('..')
sys.path.append('../..')
from UCFCrime.SpatioTemporal_Detection.NLN_STD import simple_I3D_TD_Module,load_pretrained_model,Small_I3D_TD_Module,SlowFast_STD_Module
from models.i3d import I3D
from models.resnet import generate_model
from UCFCrime.SpatioTemporal_Detection.CAM import visualize_CAM_with_clip
import os
# from models.resnet_model import generate_model
from UCFCrime.Models.build import build_model
from UCFCrime.Models_cfgs import parser
from UCFCrime.utils.checkpoint import load_checkpoint
def test_pretrained_model():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # pretrained_path='/data2/jiachang/Weakly_Supervised_VAD/models/I3D_TD/UCF_seed_0_k_10_lr_1e-05_wd_0_det_0.5_cls_0.5_l1_0.0epoch_60.pth'#,pretrained_regressor_path='/data2/jiachang/Weakly_Supervised_VAD/models/single_I3D_RGB_550'
    # model=Small_I3D_TD_Module(0.6,'/data2/jiachang/Weakly_Supervised_VAD/stored_models/model_rgb.pth').cuda().eval()
    # model_dict = model.state_dict()
    # pretrained_state_dict = torch.load(pretrained_path)['model']
    # new_dict = {k[7:]: v for k, v in pretrained_state_dict.items()}
    # model_dict.update(new_dict)
    # model.load_state_dict(model_dict)
    # model=I3D(400).cuda().eval()
    # pretrained_state_dict=torch.load('/data0/jiachang/Weakly_Supervised_VAD/stored_models/model_rgb.pth')
    # model_dict=model.state_dict()
    # new_dict={k:v for k,v in pretrained_state_dict.items() if k in model_dict.keys()}
    # model_dict.update(new_dict)
    # model.load_state_dict(model_dict)
    # load_pretrained_model(model,'/data0/jiachang/Weakly_Supervised_VAD/stored_models/model_rgb.pth')
    img_path='./imgs/cat1.jpeg'

    # model=generate_model(34)
    # pretrained_path='/data0/jiachang/Weakly_Supervised_VAD/stored_models/r3d18_K_200ep.pth'
    # model=generate_model(18,n_classes=700).cuda().eval()
    # model.load_state_dict(torch.load(pretrained_path)['state_dict'])

    # SlowFast Models
    cfg_path='../Models_cfgs/I3D_NLN_8x8_R50.yaml'
    pretrained_path='/data0/jiachang/Weakly_Supervised_VAD/stored_models/I3D_NLN_8x8_R50.pkl'
    # args=parser.parse_args()
    # args.cfg_file=cfg_path
    cfg=parser.load_config(cfg_path)
    # import pdb
    # pdb.set_trace()
    model=build_model(cfg)#.cuda().eval()
    load_checkpoint(pretrained_path,model,data_parallel=False,convert_from_caffe2=True)

    # output_model_path='/data0/jiachang/Weakly_Supervised_VAD/stored_models/{}.pth'.format(pretrained_path.split('/')[-1].split('.')[0])
    # model_state_dict=model.state_dict()
    # new_dict={k:v for k,v in model_state_dict.items() if 'head' not in k}
    # torch.save(new_dict,output_model_path)
    # frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 128.0 - 1.0
    # frames = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).cuda().float()
    # frames = frames.repeat([1, 1, 12, 1, 1])
    # frames=torch.from_numpy(np.random.random([1,3,12,224,224])*2.0-1.0).cuda().float()
    frames=torch.zeros([1,3,8,224,224]).float()#.cuda()

    # frame = (cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.-0.45)/0.225
    # frames = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).float()#.cuda()
    # frames = frames.repeat([1, 1, 8, 1, 1])

    # # frames[:,:,:,0,0]=1
    # # print(frames.shape)
    # gradcam = GradCAM(model.Regressor, grad_pp=True,index=1)
    # with torch.no_grad():
    #     scores, feat_maps = model(frames)
    # print(torch.relu(feat_maps).cpu().numpy().squeeze(0).mean(axis=(1,)).max(axis=0))
    # print(torch.relu(feat_maps).cpu().numpy().squeeze(0).mean(axis=(0,1)))
    #
    # # print(feat_maps)
    # cam_map = gradcam(feat_maps[0])
    # cam_path = './grad_cam_{}.jpg'.format(img_path.split('.')[0])
    # cam_clip = visualize_CAM_with_clip(cam_map, frames[0], (224, 224))
    # cv2.imwrite(cam_path, cam_clip)

    frames=[frames]
    with torch.no_grad():
        feat_maps,s4_maps=model(frames)
    print(s4_maps[0].shape)
    print(feat_maps[0].shape)
    # feat_map=feat_maps.mean(dim=1).mean(dim=1)[0].unsqueeze(-1)

    # feat_map=torch.var(feat_maps[0],dim=(-2,-1))
    # print(feat_map)
    # print(feat_map.max(),feat_map.min())
    # print(feat_map)
    # feat_map=feat_map-torch.min(feat_map)
    # feat_map/=feat_map.max()
    # feat_map=feat_map.cpu().numpy()
    # map=visualize_CAM(feat_map,(224,224))
    # cv2.imwrite('./grad_cam.jpg', map)


if __name__=='__main__':
    # test_GradCatcher()
    test_gradcam()
    # test_pretrained_model()