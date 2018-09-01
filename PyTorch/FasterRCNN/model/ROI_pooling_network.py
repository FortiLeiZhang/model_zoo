import torch.nn as nn
import numpy as np
import torch as t

class ROIPoolingNetwork(nn.Module):
    def __init__(self, n_class, roi_size, feat_stride):
        super(ROIPoolingNetwork, self).__init__()
        
        self.n_class = n_class
        self.roi_size = roi_size
        self.feat_stride = feat_stride
        
        self.pooling = nn.AdaptiveMaxPool2d((self.roi_size, self.roi_size))
        
    def forward(self, features, proposals):
        features = features.squeeze()
        num_proposals = proposals.shape[0]

        roi = t.zeros((num_proposals, features.shape[0], self.roi_size, self.roi_size), dtype=t.float)
        proposals = np.floor(proposals / self.feat_stride).astype(np.uint32)
        for i in range(num_proposals):
            roi_pre = t.FloatTensor(features[:, proposals[i, 1]:proposals[i, 3], proposals[i, 0]:proposals[i, 2]])
            roi[i, :, :, :] = self.pooling(roi_pre)
        return roi
