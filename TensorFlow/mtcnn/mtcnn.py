from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import random
from time import sleep

import detect_face

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ',' + str(len(self.image_paths)) + 'images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    n_classes = len(classes)
    for i in range(n_classes):
        class_name = classes[i]
        face_dir = os.path.join(path_exp, class_name)
        img_path = get_image_paths(face_dir)
        dataset.append(ImageClass(class_name, img_path))
    return dataset    

def get_image_paths(facedir):
    img_path = []
    if os.path.isdir(facedir):
        imgs = os.listdir(facedir)
        img_path = [os.path.join(facedir, img) for img in imgs]
    return img_path

def store_revision_info(output_dir, arg_string):
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, 'w') as f:
        f.write('arguments: %s\n--------------------\n' % arg_string)
        f.write('tensorflow version: %s\n--------------------\n' % tf.__version__)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int, 
                       help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
                       help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float, 
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool, 
                        help='Detect and align multiple faces per image.', default=False)
    
    return parser.parse_args(argv)

def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    store_revision_info(output_dir, ' '.join(sys.argv))
    
    dataset = get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
















