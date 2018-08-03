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
import base64

import detect_face

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

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.', default='/home/lzhang/tmp/')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.', default='/home/lzhang/mtcnn_result')
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
    
    print('Creating networks and loading parameters!')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            saver = tf.train.Saver()
            model_dir = os.path.join(args.output_dir, 'saved_model_for_serving')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            detect_face.save_mtcnn(sess, saver, model_dir)
#         with sess.as_default():
#             model_dir = '/home/lzhang/tmp/saved_model/'
#             pnet, rnet, onet = detect_face.load_mtcnn(sess, model_dir)

    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    text_file = open(bounding_boxes_filename, 'w')

    num_images_total = 0
    num_successfully_aligned = 0
    if args.random_order:
        random.shuffle(dataset)
        
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if args.random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            num_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = to_rgb(img)
                    img = img[:, :, 0:3]
                    
                    bb, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

                    num_faces = bb.shape[0]
                    if num_faces > 0:
                        det = bb[:, 0:4]
                        det_arr = []
                        h, w = np.asarray(img.shape)[0:2]
                        if num_faces > 1:
                            if args.detect_multiple_faces:
                                for i in range(num_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bb_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                offsets = np.vstack([(det[:, 2] + det[:, 0] - w) / 2, (det[:, 3] + det[:, 1] - h) / 2])
                                offset_square = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bb_size - offset_square * 2.0)
                                det_arr.append(det[index, :])
                        else:
                            det_arr.append(np.squeeze(det))
                        
                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - args.margin/2, 0)
                            bb[1] = np.maximum(det[1] - args.margin/2, 0)
                            bb[2] = np.minimum(det[2] + args.margin/2, w)
                            bb[3] = np.minimum(det[3] + args.margin/2, h)
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            num_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if args.detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            misc.imsave(output_filename_n, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
    text_file.close()
    print('Total number of images: %d' % num_images_total)
    print('Number of successfully aligned images: %d' % num_successfully_aligned)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
