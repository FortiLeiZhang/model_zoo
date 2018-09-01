import torch as t
import numpy as np
from torch import nn
import torch.nn.functional as F
from model.utils import *

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, proposal_creator_params={}):
        super(RegionProposalNetwork, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.cls_score = nn.Conv2d(mid_channels, 9 * 2, kernel_size=(1, 1), stride=(1, 1))
        self.loc_reg = nn.Conv2d(mid_channels, 9 * 4, kernel_size=(1, 1), stride=(1, 1))
        
        normal_init(self.conv, 0, 0.01)
        normal_init(self.cls_score, 0, 0.01)
        normal_init(self.loc_reg, 0, 0.01)
        
    def forward(self, features, data):
        anchor_per_center = data.anchor_per_center
        
        x = self.relu(self.conv(features))
        N, _, HH, WW = x.shape
        assert N == 1, 'only support batch_size=1'

        rpn_cls_score = self.cls_score(x) # (1, 18, H, W)
        rpn_loc_reg = self.loc_reg(x) # (1, 36, H, W)
        
        rpn_loc_reg = rpn_loc_reg.permute(0, 2, 3, 1).contiguous().view(N, -1, 4).squeeze()
        
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous()
        rpn_cls_score = rpn_cls_score.view(N, HH, WW, anchor_per_center, 2)
        rpn_cls_score = F.softmax(rpn_cls_score, dim=4)

        rpn_foreground_score = rpn_cls_score[:, :, :, :, 1].contiguous()
        rpn_foreground_score = rpn_foreground_score.view(N, -1).squeeze()
        rpn_cls_score = rpn_cls_score.view(N, -1, 2).squeeze()
        
        return rpn_foreground_score, rpn_loc_reg, rpn_cls_score
