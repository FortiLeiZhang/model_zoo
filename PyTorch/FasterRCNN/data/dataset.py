import os
import numpy as np
import random
import cv2
import xml.etree.ElementTree as ET

from data import data_util
from torch.utils.data import Dataset

from model.anchor_generator import AnchorGenerator, AnchorTargetGenerator

class VOCBboxDataSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset_split = self.args.dataset_split
        self.min_size = args.min_size
        self.max_size = args.max_size
        self.use_difficult = self.args.use_difficult
        self.return_difficult = self.args.return_difficult
        
        self.dataset_dir = self.args.dataset_dir
        self.id_file_dir = os.path.join(self.dataset_dir, 'ImageSets/Main/')
        self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
        self.anno_dir = os.path.join(self.dataset_dir, 'Annotations')

        self.id_file_name = '{0}.txt'.format(self.dataset_split)
        self.id_file = os.path.join(self.id_file_dir, self.id_file_name)
        self.id_list = [line.split()[0].strip() for line in open(self.id_file, 'r')]

        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        voc_data = VOCBboxData(self.args, idx)
        return voc_data()

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

class VOCBboxData(VOCBboxDataSet):
    def __init__(self, args, idx):
        super(VOCBboxData, self).__init__(args)
        self.use_data_aug = args.use_data_aug
        self.random_hflip_ratio = args.random_hflip_ratio
        
        self.args = args
        self.idx = idx
        self.img_id = self.id_list[self.idx]
        self.img_file = os.path.join(self.img_dir, self.img_id + '.jpg')
        
        self.img_data = None
        self.ori_img_size = (0, 0)
        self.img_size = (0, 0)
        self.scale = 0.0
        
        self.anno_file = os.path.join(self.anno_dir, self.img_id + '.xml')
        self.anno = ET.parse(self.anno_file)
        
        self.bbox = list()
        self.label = list()
        self.difficult = list()
        
        self.anchor_scale = [8, 16, 32]
        self.anchor_ratio = [0.5, 1, 2]
        self.anchor_per_center = int(len(self.anchor_scale)) * int(len(self.anchor_ratio))
        self.anchor_generator = AnchorGenerator()
        self.anchor_target_generator = AnchorTargetGenerator()
        self.num_anchors = 0
        self.num_centers = 0
        self.anchors = None
        self.anchor_target = None

    def __call__(self):
        for obj in self.anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            self.difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            self.bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            self.label.append(VOC_BBOX_LABEL_NAMES.index(name))
        self.bbox = np.stack(self.bbox).astype(np.float32)
        self.label = np.stack(self.label).astype(np.int32)
        self.difficult = np.array(self.difficult, dtype=np.bool).astype(np.uint8)
        
        self.img_data = cv2.imread(self.img_file)
        self.ori_img_size = (self.img_data.shape[0], self.img_data.shape[1])
        
#         if self.args.debug:
#             img_data1 = self.img_data.copy()
#             for i in range(self.bbox.shape[0]):
#                 data_util.draw_bounding_box_on_image(img_data1, self.bbox[i, :])
#             cv2.imwrite(os.path.join('/home/lzhang/tmp/', self.img_id + '_ori.jpg'), img_data1)

        self.img_data, self.scale = data_util.resize_img(self.img_data, self.min_size, self.max_size)
        self.img_size = (self.img_data.shape[0], self.img_data.shape[1])
        self.bbox = data_util.resize_bbox(self.bbox, self.scale)
        if self.use_data_aug:
            p_rnd = random.random()
            if p_rnd < self.random_hflip_ratio:
                self.img_data = data_util.hflip_img(self.img_data)
                self.bbox = data_util.hflip_bbox(self.bbox, (self.img_data.shape[0], self.img_data.shape[1]))
                
#         if self.args.debug:
#             img_data2 = self.img_data.copy()
#             for i in range(self.bbox.shape[0]):
#                 data_util.draw_bounding_box_on_image(img_data2, self.bbox[i, :])
#             cv2.imwrite(os.path.join('/home/lzhang/tmp/', self.img_id + '_flip.jpg'), img_data2)

        self.anchors = self.anchor_generator(self.img_size)
        self.num_anchors = self.anchors.shape[0]
        self.num_centers = int(self.num_anchors / self.anchor_per_center)
        if self.args.training:
            self.anchor_target = self.anchor_target_generator(self.img_size, self.anchors, self.bbox)
        return self
