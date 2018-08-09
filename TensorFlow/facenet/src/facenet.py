from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
import math
from six import iteritems

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))

def store_revision_info(src_path, output_dir, arg_string):
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror

    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    num_classes = len(classes)
    for i in range(num_classes):
        class_name = classes[i]
        face_dir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(face_dir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths(face_dir):
    image_paths = []
    if os.path.isdir(face_dir):
        images = os.listdir(face_dir)
        image_paths = [os.path.join(face_dir, img) for img in images]
    return image_paths
        
class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
    def __len__(self):
        return len(self.image_paths)
    
def split_dataset(dataset, split_ratio, min_val_images_per_class, mode):
    if mode == 'SPLIT_CLASSES':
        num_classes = len(dataset)
        class_index = np.arange(num_classes)
        np.random.shuffle(class_index)
        split = int(round(num_classes * (1 - split_ratio)))
        train_set = [dataset[i] for i in class_index[0:split]]
        val_set = [dataset[i] for i in class_index[split:-1]]
    elif mode == 'SPLIT_IMAGES':
        train_set = []
        val_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            num_images_in_class = len(paths)
            split = int(math.floor(num_images_in_class * (1 - split_ratio)))
            if split == num_images_in_class:
                split = num_images_in_class - 1
            if split >= min_val_images_per_class and num_images_in_class - split >= 1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                val_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, val_set    
    
def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        