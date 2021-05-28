import torch
from torch import nn
import torch.nn.functional as F
import copy
class Self_Guided_Attention_Branch_Module(nn.Module):
    def __init__(self,in_channels,expand_k,out_t_channels):
        super(Self_Guided_Attention_Branch_Module, self).__init__()
        self.in_channels = in_channels
        self.num_headers = 2 * expand_k

        if out_t_channels==4:
            self.Conv_Atten=nn.Sequential(nn.Conv3d(in_channels,self.in_channels,kernel_size=(3,3,3),stride=(1,2,2),padding=(1,1,1)),
                                          nn.ReLU(),
                                          )
        elif out_t_channels==2:
            self.Conv_Atten=nn.Sequential(nn.Conv3d(in_channels,self.in_channels,kernel_size=(3,3,3),stride=(1,2,2),padding=(0,1,1)),
                                            nn.ReLU(),
                                            )

        # we have to copy the weight of
        self.Att_1=nn.Sequential(nn.Conv3d(in_channels,self.num_headers,kernel_size=(1,1,1)),nn.ReLU())
        self.Att_2=nn.Conv3d(self.num_headers,self.num_headers,kernel_size=(1,1,1))
        self.GAP=nn.AdaptiveAvgPool3d(1)
        self.Softmax=nn.Softmax(dim=-1)

        self.Att_3=nn.Sequential(nn.Conv3d(self.num_headers,1,kernel_size=(out_t_channels,1,1)),nn.Sigmoid())

    def forward(self,x):
        b,c,t,h,w =x.shape
        feat_map=self.Conv_Atten(x)
        feat_map=self.Att_1(feat_map)
        # att_scores=self.Softmax(self.GAP(self.Att_2(feat_map)).squeeze(-1).squeeze(-1).squeeze(-1).view(feat_map.shape[0],2,-1).mean(dim=-1))
        att_scores=self.GAP(self.Att_2(feat_map)).squeeze(-1).squeeze(-1).squeeze(-1).view(feat_map.shape[0],2,-1).mean(dim=-1)
        att_map=self.Att_3(feat_map)
        return att_map,att_scores,feat_map
