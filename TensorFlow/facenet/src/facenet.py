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

RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)

def create_input_pipeline(input_q, image_size, num_preprocess_threads, batch_size_placeholder):
    images_and_labels_list = []
    for _ in range(num_preprocess_threads):
        filenames, label, control = input_q.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
            
            image = tf.cond(get_control_flag(control[0], RANDOM_CROP), 
                           lambda : tf.random_crop(image, image_size + (3, )), 
                           lambda : tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))

            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP), 
                           lambda : tf.image.random_flip_left_right(image), 
                           lambda : tf.identity(image))

            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION), 
                           lambda : (tf.cast(image, tf.float32) - 127.5) / 128.0, 
                           lambda : tf.image.per_image_standardization(image))

            image = tf.cond(get_control_flag(control[0], FLIP), 
                           lambda : tf.image.flip_left_right(image), 
                           lambda : tf.identity(image))
            
            image.set_shape(image_size + (3, ))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder,
        shapes=[image_size + (3, ), ()], enqueue_many=True, 
        capacity=400 * num_preprocess_threads, allow_smaller_final_batch=True)

    return image_batch, label_batch    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        