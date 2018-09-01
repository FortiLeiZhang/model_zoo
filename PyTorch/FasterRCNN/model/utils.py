import numpy as np

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
        
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    
    top_left = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    bottom_right = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_intersection = np.prod(bottom_right - top_left, axis=2) * (top_left < bottom_right).all(axis=2)

    return area_intersection / (area_a[:, None] + area_b - area_intersection)
    
def bbox2reg(src_bbox, dst_bbox):
    w = src_bbox[:, 2] - src_bbox[:, 0] + 1.0
    h = src_bbox[:, 3] - src_bbox[:, 1] + 1.0
    x_ctr = src_bbox[:, 0] + 0.5 * w
    y_ctr = src_bbox[:, 1] + 0.5 * h
    
    base_w = dst_bbox[:, 2] - dst_bbox[:, 0] + 1.0
    base_h = dst_bbox[:, 3] - dst_bbox[:, 1] + 1.0
    base_x_ctr = dst_bbox[:, 0] + 0.5 * base_w
    base_y_ctr = dst_bbox[:, 1] + 0.5 * base_h

    eps = np.finfo(h.dtype).eps
    w = np.maximum(w, eps)
    h = np.maximum(h, eps)
    
    dx = (base_x_ctr - x_ctr) / w
    dy = (base_y_ctr - y_ctr) / h
    dw = np.log(base_w / w)
    dh = np.log(base_h / h)

    return np.vstack((dx, dy, dw, dh)).transpose()

def reg2bbox(src_bbox, reg):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=reg.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    
    src_width = src_bbox[:, 2] - src_bbox[:, 0] + 1.0
    src_height = src_bbox[:, 3] - src_bbox[:, 1] + 1.0
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx = reg[:, 0::4]
    dy = reg[:, 1::4]
    dw = reg[:, 2::4]
    dh = reg[:, 3::4]

    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]

    dst_bbox = np.zeros(reg.shape, dtype=reg.dtype)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w - 1.0
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h - 1.0

    return dst_bbox

def unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()

def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if bbox.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    
    num_bbox = bbox.shape[0]
    
    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = np.arange(num_bbox, dtype=np.int32)    
    
    sorted_bbox = bbox[order, :]
    keep_idx = []
    
    while len(order) >= 1:
        i = order[0]
        keep_idx.append(i)
        
        if len(order) == 1:
            break

        order = order[1:]       
        keep_bbox = np.expand_dims(bbox[i, :], 0)
        rest_bbox = bbox[order, :]
        
        ious = bbox_iou(keep_bbox, rest_bbox).squeeze()
        idx = np.where(ious <= thresh)[0]
        order = order[idx]

    return keep_idx

def test_nms():
    np.random.seed(1)   # keep fixed
    num_rois = 6000
    minxy = np.random.randint(50, 145, size=(num_rois ,2))
    maxxy = np.random.randint(150, 200, size=(num_rois ,2))

    score = 0.8 * np.random.random_sample((num_rois, 1)) + 0.2
    order = score.ravel().argsort()[::-1]

    boxes_new = np.concatenate((minxy, maxxy), axis=1).astype(np.float32)
    boxes_new = boxes_new[order, :]

    keep = non_maximum_suppression(boxes_new, thresh=0.7)
    print(len(keep)) # 545