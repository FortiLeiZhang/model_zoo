import torch as t
import numpy as np
from torch import nn
import torch.nn.functional as F

from model.feature_extraction_network import FeatureExtractionNetwork
from model.region_proposal_network import RegionProposalNetwork
from model.proposal_generator import ProposalGenerator
from model.ROI_pooling_network import ROIPoolingNetwork
from model.utils import *

class FasterRCNNVGG16(nn.Module):
    def __init__(self, n_fg_class=20):
        super(FasterRCNNVGG16, self).__init__()
        
        self.feat_stride = 16
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.n_fg_class = n_fg_class

        self.vgg16 = FeatureExtractionNetwork()
        self.extractor = self.vgg16.features
        self.classifier = self.vgg16.classifier

        self.rpn = RegionProposalNetwork()
        self.proposal_generator = ProposalGenerator(self)
        
        self.roi_pooling = ROIPoolingNetwork(
            n_class=self.n_fg_class + 1,
            roi_size = 7,
            feat_stride = self.feat_stride
        )
        
        self.final_cls = nn.Linear(4096, self.n_fg_class + 1)
        self.final_reg = nn.Linear(4096, (self.n_fg_class + 1) * 4)
        
        normal_init(self.final_cls, 0, 0.001)
        normal_init(self.final_reg, 0, 0.01)

    def forward(self, data):
        img = data.img_data
        bbox = data.bbox

        height, width = img.shape[0], img.shape[1]
        img_size = (height, width)
        
        img_tensor = t.from_numpy(np.expand_dims(img, axis=0).transpose((0, 3, 1, 2))).type(t.float)
        features = self.extractor(img_tensor)
        
        rpn_foreground_score, rpn_loc_reg, rpn_cls_score = self.rpn(features, data)
        
        proposals = self.proposal_generator(rpn_loc_reg, rpn_foreground_score, data)
        
        roi = self.roi_pooling(features, proposals)
        num_roi = roi.shape[0]
        roi = roi.view(num_roi, -1)
        h = self.classifier(roi)
        
        final_cls = F.softmax(self.final_cls(h), dim=1)
        final_reg = self.final_reg(h)
        
        t.set_printoptions(threshold=5000, edgeitems=10)
        print(final_cls.shape)
        print(final_reg.shape)

        
#     def suppress(self, raw_cls, raw_reg):
#         bbox = []
#         label = []
#         score = []
        
#         for i in range(1, self.n_fg_class + 1):
            
        
        
        
        
        
        
        
        
        
        
        
        
