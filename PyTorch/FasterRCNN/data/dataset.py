import os
import numpy as np
import random
from PIL import Image
import xml.etree.ElementTree as ET

from data import data_util
from torch.utils.data import Dataset

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
        self.id_list = [line.strip() for line in open(self.id_file, 'r')]

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
        self.img_PIL = None
        
        self.anno_file = os.path.join(self.anno_dir, self.img_id + '.xml')
        self.anno = ET.parse(self.anno_file)
        
        self.bbox = list()
        self.label = list()
        self.difficult = list()
        
    def __call__(self):
        for obj in self.anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            self.difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            self.bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            self.label.append(VOC_BBOX_LABEL_NAMES.index(name))
        self.bbox = np.stack(self.bbox).astype(np.float32)
        self.label = np.stack(self.label).astype(np.int32)
        self.difficult = np.array(self.difficult, dtype=np.bool).astype(np.uint8)
        
        self.img_PIL = Image.open(self.img_file).convert('RGB')

        if self.args.debug:
            img_PIL1 = self.img_PIL.copy()
            for i in range(self.bbox.shape[0]):
                data_util.draw_bounding_box_on_image(img_PIL1, self.bbox[i, :])
            img_PIL1.save(os.path.join('/home/lzhang/tmp/', self.img_id + '_ori.jpg'))
        
        W, H = self.img_PIL.size
        self.img_PIL = data_util.resize_img(self.img_PIL, self.min_size, self.max_size)
        o_W, o_H = self.img_PIL.size
        self.bbox = data_util.resize_bbox(self.bbox, (W, H), (o_W, o_H))
        if self.use_data_aug:
            p_rnd = random.random()
            if p_rnd < self.random_hflip_ratio:
                self.img_PIL = data_util.hflip_img(self.img_PIL)
                self.bbox = data_util.hflip_bbox(self.bbox, (o_W, o_H))

        if self.args.debug:
            img_PIL2 = self.img_PIL.copy()
            for i in range(self.bbox.shape[0]):
                data_util.draw_bounding_box_on_image(img_PIL2, self.bbox[i, :])
            img_PIL2.save(os.path.join('/home/lzhang/tmp/', self.img_id + '_flip.jpg'))        

        return self.img_PIL, self.bbox, self.label, self.difficult
