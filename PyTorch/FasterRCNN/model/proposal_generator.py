import numpy as np

from model.utils import *

class ProposalGenerator(object):
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, reg, score, data):
        anchor = data.anchors
        scale = data.scale
        (height, width) = data.img_size

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        
        roi = reg2bbox(anchor, reg.data.numpy())
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, width)
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, height)

        min_size = self.min_size
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = score.data.numpy().ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]

        keep = non_maximum_suppression(roi, thresh=self.nms_thresh)

        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]

        return roi
