from __future__ import division

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, 
                loc_normal_mean = (0., 0., 0., 0.),
                loc_normal_std = (0.1, 0.1, 0.2, 0.,2)):
        super(FasterRCNN, self),__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        
        self.loc_normal_mean = loc_normal_mean
        self.loc_normal_std = loc_normal_std
        self.use_preset('evaluate')
        
    @property
    def n_class(self):
        return self.head.n_class
    
    def forward(self, x, scale=1.0):
        img_size = x.shape[2:]
        
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices
    
    def use_preset(self, preset):
        if preset = 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset = 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')
    
    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        
        for i in range(1, self.n_class):
            cls_bbox = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, i, :]
            prob = raw_prob[:, i]
            mask = prob > self.score_thresh
            cls_bbox = cls_bbox[mask]
            prob = prob[maks]
            keep = non_maximum_suppression(np.array(cls_bbox), self.num_thresh, prob)
            keep = np.asarray(keep)
            bbox.append(cls_bbox[keep])
            label.append((i - 1) * np.ones((len(keep), )))
            score.append(prob[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        
        return bbox, label, score
    
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = []
        lables = []
        scores = []
        for img, size in zip(prepared_imgs, sizes):
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
        