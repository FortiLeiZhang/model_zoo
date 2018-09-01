from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import time

from torch.utils.data import DataLoader

from data.dataset import VOCBboxDataSet
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.anchor_generator import AnchorGenerator, AnchorTargetGenerator

def parse_args(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_dir', default='/home/lzhang/pascal2/VOC2007/', type=str, help='training dataset dir')
    parser.add_argument('--dataset_split', default='train_test', type=str, help='training dataset split')
    parser.add_argument('--use_difficult', action='store_true', help='whether use difficult bbox')
    parser.add_argument('--return_difficult', action='store_true', help='whether return difficult bbox')
    parser.add_argument('--use_data_aug', action='store_true', help='whether use data augmentation')
    parser.add_argument('--random_hflip_ratio', default=0.5, type=float, help='ratio for horizontal flipping image')
    
    parser.add_argument('--net', default='vgg16', type=str, help='vgg16, res101')
    parser.add_argument('--min_size', default=600, type=int, help='min image resize')
    parser.add_argument('--max_size', default=1000, type=int, help='max image resize')
    parser.add_argument('--num_workers', default=8, type=int, help='number of worker to load data')
    
    parser.add_argument('--rpn_sigma', default=3, type=int, help='rpn sigma for l1_smooth_loss')
    parser.add_argument('--roi_sigma', default=1, type=int, help='roi sigma for l1_smooth_loss')
    
    parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
    parser.add_argument('--lr_decay_step', default=5, type=int, help='epoch to do learning rate decay')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='learning rate decay ratio')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay ratio')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs to train')
    parser.add_argument('--optimizer', default="sgd", type=str, help='training optimizer')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')    
    parser.add_argument('--cuda', action='store_true', help='whether use CUDA')
    
    parser.add_argument('--visdom_env', default='faster_rcnn', type=str, help='visdom env')
    parser.add_argument('--visdom_port', default='8097', type=str, help='visdom port')
    
    parser.add_argument('--plot_every', default=100, type=int, help='number of iterations to plot')
    parser.add_argument('--save_ckpt_every', default=10000, type=int, help='number of iterations to save checkpoint.')
    parser.add_argument('--save_dir', default="/home/lzhang/pytorch/models", type=str, help='directory to save models')

    parser.add_argument('--debug', action='store_true', help='if print debug msg')
    parser.add_argument('--training', action='store_true', help='if training')
    print(parser.parse_args())
    return parser.parse_args()

def train(args):   
    dataset = VOCBboxDataSet(args)
    for i in range(20):
        dataset[i]
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=args.num_workers)
    
#     faster_rcnn = FasterRCNNVGG16()
    
if __name__ == '__main__':
    train(parse_args(sys.argv[1:]))
    
# python ./train.py --debug --use_data_aug --random_hflip_ratio=1.0