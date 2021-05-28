import torch
from torch import nn
from utils.utils import weights_normal_init

class Simple_Regressor(nn.Module):
    def __init__(self,input_feature_dim,dropout_rate=0.6):
        super(Simple_Regressor, self).__init__()
        self.regressor=nn.Sequential(nn.Linear(input_feature_dim,512),nn.ReLU(),nn.Dropout(dropout_rate),
                                     nn.Linear(512,32),nn.Dropout(dropout_rate),
                                     nn.Linear(32,1),nn.Sigmoid())

        weights_normal_init(self.regressor)

    def forward(self, x):
        x=x.view([-1,x.shape[-1]])
        logits=self.regressor(x)
        return logits