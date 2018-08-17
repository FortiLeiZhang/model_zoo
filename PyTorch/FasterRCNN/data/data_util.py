import numpy as np
from PIL import Image
from PIL import ImageDraw
import random
from torchvision import transforms as T

def draw_bounding_box_on_image(image, box, color='red', thickness=4):
    draw = ImageDraw.Draw(image)

    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

def hflip_img(img):
    return T.functional.hflip(img)

def resize_img(img, min_size=600, max_size=1000):
    img = img.copy()
    W, H = img.size
    scale1 = min_size / min(W, H)
    scale2 = max_size / max(W, H)
    scale = min(scale1, scale2)
    img.resize((int(W * scale), int(H * scale)), Image.ANTIALIAS)
    return img
    
def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    x_scale = float(out_size[0]) / in_size[0]
    y_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def flip_bbox(bbox, size):
    W, H = size
    bbox = bbox.copy()
    x_max = W - bbox[:, 1]
    x_min = W - bbox[:, 3]
    bbox[:, 1] = x_min
    bbox[:, 3] = x_max
    return bbox
