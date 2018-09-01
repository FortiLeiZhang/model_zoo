import cv2
from torchvision import transforms as T

def draw_bounding_box_on_image(image, box, color=(0, 0, 255), thickness=4):
    ymin, xmin, ymax, xmax = box
    cv2.rectangle(image, (xmin, ymin),(xmax, ymax), color=color, thickness=thickness)

def hflip_img(img):
    return cv2.flip(img, 1)

def resize_img(img, min_size=600, max_size=1000):
    img = img.copy()
    H, W, _ = img.shape
    scale1 = min_size / min(W, H)
    scale2 = max_size / max(W, H)
    scale = min(scale1, scale2)
    return cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR), scale
    
def resize_bbox(bbox, scale):
    bbox = bbox.copy()
    bbox[:, 0] = scale * bbox[:, 0]
    bbox[:, 2] = scale * bbox[:, 2]
    bbox[:, 1] = scale * bbox[:, 1]
    bbox[:, 3] = scale * bbox[:, 3]
    return bbox

def hflip_bbox(bbox, size):
    H, W = size
    bbox = bbox.copy()
    x_max = W - bbox[:, 1]
    x_min = W - bbox[:, 3]
    bbox[:, 1] = x_min
    bbox[:, 3] = x_max
    return bbox
