import torch
from torch import nn
from models.Attention import *

def load_c3d_pretrained_model(net,checkpoint_path,name=None):
    checkpoint = torch.load(checkpoint_path)
    state_dict = net.state_dict()
    base_dict = {}
    checkpoint_keys = checkpoint.keys()
    if name==None:
        for k, v in state_dict.items():
            for _k in checkpoint_keys:

                if k in _k:
                    base_dict[k] = checkpoint[_k]
    else:
        if name=='fc6':
            base_dict['0.weight']=checkpoint['backbone.fc6.weight']
    #         base_dict['0.bias']=checkpoint['backbone.fc6.bias']
    # import pdb
    # pdb.set_trace()
    state_dict.update(base_dict)
    net.load_state_dict(state_dict)
    print('model load pretrained weights')
    
class C3DBackbone(nn.Module):
    def __init__(self):
        super(C3DBackbone, self).__init__()
        # 112
        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 56
        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 28
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 14
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 7
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        # self.pool5=nn.AdaptiveAvgPool3d(1)
        # self.fc6 = nn.Linear(8192, 4096)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.relu(self.conv1a(x))
        x = self.pool1(x)
        x = self.relu(self.conv2a(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        out_4=x

        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        # x = self.pool5(x)

        # x = x.view(-1, 8192)
        # x = self.relu(self.fc6(x))
        return x,out_4
    
    
class C3D_SGA_STD(nn.Module):
    def __init__(self,dropout_rate,expand_k,freeze_backbone,freeze_blocks,pretrained_backbone=False,pretrained_path=None):
        super(C3D_SGA_STD, self).__init__()
        self.backbone=C3DBackbone()
        self.Regressor=nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(512,2))

        self.GAP=nn.AdaptiveAvgPool3d(1)
        self.freeze_backbone = freeze_backbone
        if freeze_blocks==None:
            self.freeze_blocks=['conv1a','conv2a','conv3a','conv3b','conv4a','conv4b','conv5a','conv5b']
        else:
            self.freeze_blocks = freeze_blocks
        self.Softmax=nn.Softmax(dim=-1)
        self.pretrained_backbone=pretrained_backbone

        self.Conv_Atten=Self_Guided_Attention_Branch_Module(512,expand_k,out_t_channels=2)

        if self.pretrained_backbone and pretrained_path!=None:
            load_c3d_pretrained_model(self.backbone,pretrained_path)

    def freeze_part_model(self):
        if self.freeze_backbone:
            for name,p in self.backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=False

        else:
            for name,p in self.backbone.named_parameters():
                p.requires_grad=True


    def train(self,mode=True):
        super(C3D_SGA_STD, self).train(mode)
        if self.freeze_backbone:
            self.freeze_part_model()
        return self


    def forward(self,x,act=True,extract=False):
        feat_map,feat_map_4=self.backbone(x)
        atten_map,atten_logits,att_feat_map=self.Conv_Atten(feat_map_4)
        atten_feat_map=atten_map*feat_map
        # feat_map=torch.cat([feat_map,atten_feat_map],dim=1)
        feat_map=feat_map+atten_feat_map

        feat=self.GAP(feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
        # feat=self.pool5(feat_map).view(-1,8192)
        # feat=self.fc6(feat)

        logits=self.Regressor(feat)
        if act:
            logits=self.Softmax(logits)
            atten_logits=self.Softmax(atten_logits)
        if not extract:
            return logits,feat_map,atten_logits,atten_map
        else:
            return logits,feat_map,atten_logits,atten_map,att_feat_map
