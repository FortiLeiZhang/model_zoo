from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
from PIL import Image


model_dir = '/home/lzhang/facenet/20180408-102900'
image_dir = '/home/lzhang/tmp/test_160'

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def facenet_client():
    img_list = []
    img_name_list = []
    for file in os.listdir(image_dir):
        img_name = os.path.join(image_dir, file)
        img_name_list.append(img_name)
        img = Image.open(img_name)
#         img = img.resize((160, 160))
        img = np.array(img)
        img_p = prewhiten(img)
        img_list.append(img_p)
    images = np.stack(img_list)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = images.shape[0]
            
#             for i in range(nrof_images):
#                 print('Image name: %s' %img_name_list[i])
#                 print('Embedding: %s' %emb[i])

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')
            
            
# def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

#     minsize = 20 # minimum size of face
#     threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
#     factor = 0.709 # scale factor
    
#     print('Creating networks and loading parameters')
#     with tf.Graph().as_default():
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#         with sess.as_default():
#             pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
#     tmp_image_paths=copy.copy(image_paths)
#     img_list = []
#     for image in tmp_image_paths:
#         img = misc.imread(os.path.expanduser(image), mode='RGB')
#         img_size = np.asarray(img.shape)[0:2]
#         bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
#         if len(bounding_boxes) < 1:
#           image_paths.remove(image)
#           print("can't detect face, remove ", image)
#           continue
#         det = np.squeeze(bounding_boxes[0,0:4])
#         bb = np.zeros(4, dtype=np.int32)
#         bb[0] = np.maximum(det[0]-margin/2, 0)
#         bb[1] = np.maximum(det[1]-margin/2, 0)
#         bb[2] = np.minimum(det[2]+margin/2, img_size[1])
#         bb[3] = np.minimum(det[3]+margin/2, img_size[0])
#         cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
#         aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#         prewhitened = facenet.prewhiten(aligned)
#         img_list.append(prewhitened)
#     images = np.stack(img_list)
#     return images

# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('model', type=str, 
#         help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
#     parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
#     parser.add_argument('--image_size', type=int,
#         help='Image size (height, width) in pixels.', default=160)
#     parser.add_argument('--margin', type=int,
#         help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
#     parser.add_argument('--gpu_memory_fraction', type=float,
#         help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
#     return parser.parse_args(argv)

if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))
    facenet_client()
