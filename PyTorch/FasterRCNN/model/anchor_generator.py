import torch as t
import numpy as np
from torch import nn

from model.utils import *

def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * np.asarray(scales)
    hs = h * np.asarray(scales)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors  

def _get_inside_index(anchors, H, W):
    idx_inside = np.where(
        (anchors[:, 0] >= 0) & 
        (anchors[:, 1] >= 0) & 
        (anchors[:, 2] < W) & 
        (anchors[:, 3] < H)
    )[0]
    return idx_inside

class AnchorGenerator(object):
    # anchor: [x_{min}, y_{min}, x_{max}, y_{max}]
    def __init__(self, feat_stride=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
        self.feat_stride = feat_stride
        self.ratios = ratios
        self.scales = scales
        
        self.meta_anchor = np.array([1, 1, self.feat_stride, self.feat_stride]) - 1
        ratio_anchors = _ratio_enum(self.meta_anchor, self.ratios)
        self.base_anchor = np.vstack([_scale_enum(ratio_anchors[i, :], self.scales) for i in range(ratio_anchors.shape[0])])        

        self.anchors = None
        self.box_per_anchor = int(len(ratios)) * int(len(scales))
        self.num_anchors = 0
        
    def __call__(self, img_size):
        (self.height, self.width) = img_size
        hh = self.height // 2 // 2 // 2 // 2
        ww = self.width // 2 // 2 // 2 // 2

        shift_x = np.arange(0, ww * self.feat_stride, self.feat_stride) 
        shift_y = np.arange(0, hh * self.feat_stride, self.feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        num_center = shifts.shape[0]
        anchors_per_center = self.base_anchor.shape[0]
        self.anchors = (self.base_anchor.reshape((1, anchors_per_center, 4)) + 
                        shifts.reshape((1, num_center, 4)).transpose((1, 0, 2)))
        self.anchors = self.anchors.reshape((num_center * anchors_per_center, 4))
        self.num_anchors = self.anchors[0]
        return self.anchors

class AnchorTargetGenerator(object):
    def __init__(self, num_samples=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.num_samples = num_samples
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
        
        self.height = 0
        self.width = 0
        self.anchor = None
        self.bbox = None

    def __call__(self, img_size, anchor, bbox):
        (self.height, self.width) = img_size
        self.anchor = anchor
        self.bbox = bbox

        inside_index = _get_inside_index(self.anchor, self.height, self.width)
        inside_anchor = self.anchor[inside_index]
        
        per_anchor_bbox_idx, label = self.create_label(inside_anchor)
        reg_target = bbox2reg(inside_anchor[label != -1], self.bbox[per_anchor_bbox_idx[label != -1]])

        reg_target = unmap(reg_target, self.anchor.shape[0], inside_index[label != -1], fill=0)       
        label = unmap(label, self.anchor.shape[0], inside_index, fill=-1)
        return label, reg_target
    
    def filter_ious(self, anchor, bbox):
        ious = bbox_iou(anchor, bbox)

        per_anchor_bbox_idx = ious.argmax(axis=1) # 每个 anchor 与第几个 bbox 的 IoU 最大
        per_anchor_max_ious = ious[np.arange(anchor.shape[0]), per_anchor_bbox_idx] # 最大值是多少
        
        per_bbox_anchor_idx = ious.argmax(axis=0) # 每个 bbox 与第几个 anchor 的 IoU 最大
        per_bbox_max_ious = ious[per_bbox_anchor_idx, np.arange(ious.shape[1])] # 最大值是多少
        per_bbox_anchor_idx = np.where(ious == per_bbox_max_ious)[0] # 所有最大值的 idx
        
        return per_anchor_bbox_idx, per_anchor_max_ious, per_bbox_anchor_idx

    def create_label(self, inside_anchor):
        label = np.empty((inside_anchor.shape[0], ), dtype=np.float32)
        label.fill(-1)
       
        per_anchor_bbox_idx, per_anchor_max_ious, per_bbox_anchor_idx = self.filter_ious(inside_anchor, self.bbox)

        label[per_anchor_max_ious < self.neg_iou_thresh] = 0
        label[per_bbox_anchor_idx] = 1
        label[per_anchor_max_ious >= self.pos_iou_thresh] = 1
        
        num_positive = int(self.pos_ratio * self.num_samples)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > num_positive:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - num_positive), replace=False)
            label[disable_index] = -1
        
        num_negtive = self.num_samples - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > num_negtive:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - num_negtive), replace=False)
            label[disable_index] = -1
        
        return per_anchor_bbox_idx, label
