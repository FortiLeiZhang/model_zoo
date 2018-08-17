import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

from data import data_util

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

class VOCBboxDataset:
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False):
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        self.ids = [line.strip() for line in open(id_list_file, 'r')]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', img_id + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)
        
        img_file = os.path.join(self.data_dir, 'JPEGImages', img_id + '.jpg')
        img = Image.open(img_file).convert('RGB')

        for i in range(bbox.shape[0]):
            data_util.draw_bounding_box_on_image(img, bbox[i, :])
        img.save(os.path.join('/home/lzhang/tmp/', img_id + '_ori.jpg'))
        img = np.asarray(img, dtype=np.uint8)
        
        return img, bbox, label, difficult
