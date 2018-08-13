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


model_dir = '/home/lzhang/model_zoo/TensorFlow/facenet/models/my'
image_dir = '/home/lzhang/tmp/0000045_160'

def facenet_client():
    img_list = []
    
    for file in os.listdir(image_dir):
        with open(os.path.join(image_dir, file), 'rb') as f:
            img = f.read()
            img_list.append(img)
    images = np.stack(img_list)

#     for file in os.listdir(image_dir):
#         img = Image.open(os.path.join(image_dir, file), 'r')
#         img = np.array(img)
#         tf.cast(img, tf.float32)
#         img_list.append(img)
#     images = np.stack(img_list)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            feed_dict = { images_placeholder: images }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            print(emb.shape)
            
            nrof_images = images.shape[0]

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
                
if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))
    facenet_client()