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
from visualization.CAM import visualize_CAM
def test_gradcam():
    model=Simple_Regressor(1024).cuda().eval()
    old_state_dict=torch.load('/data2/jiachang/Weakly_Supervised_VAD/models/single_I3D_RGB_550')
    model.load_state_dict({'r'+k[1:]:v for k,v in old_state_dict.items()})
    feature_maps=torch.from_numpy(np.random.random([1024,2,8,10])).float().cuda()
    gradcam=GradCAM(model,grad_pp=False)
    cam=gradcam(feature_maps)
    print(cam.shape)
    visualize_CAM(cam,(320,240))


if __name__=='__main__':
    # test_GradCatcher()
    test_gradcam()
    # test_pretrained_model()
