from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import time
import tensorflow as tf
import numpy as np
import importlib
import argparse
import math
import re

import facenet

def load_model(sess, model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = facenet.get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(sess, os.path.join(model_exp, ckpt_file))

def decode_encoded_image_string_tensor(encoded_image_string_tensor):
    image_tensor = tf.image.decode_image(encoded_image_string_tensor, channels=3)
    image_tensor.set_shape((None, None, 3))
    return image_tensor

def save_variables_and_metagraph(sess, saver, model_dir, model_name):
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)

def main(args):
    load_from_model_dir = args.load_from
    save_to_model_dir = args.save_to
    model_name = args.model

    network = importlib.import_module(model_name)
    
    graph1 = tf.Graph()
    with graph1.as_default():
        sess1 = tf.Session()
        load_model(sess1, load_from_model_dir)

    with tf.Graph().as_default():
        img_str_placeholder = tf.placeholder(dtype=tf.string, shape=(None, ), name='input')
        img_tensor = tf.map_fn(decode_encoded_image_string_tensor, elems=img_str_placeholder, dtype=tf.uint8, back_prop=False)
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor.set_shape((None, 160, 160, 3))
        
#         img_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 160, 160, 3), name='input')
    
        prelogits, _ = network.inference(img_tensor, 1.0, phase_train=False, bottleneck_layer_size=512, weight_decay=0.0)       
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            i = 0
            for var_name in tf.trainable_variables():
                copy_tensor = graph1.get_tensor_by_name(var_name.name)
                save_tensor = tf.get_default_graph().get_tensor_by_name(var_name.name)
                weight = sess1.run(copy_tensor)
                sess.run(var_name.assign(weight))
                i += 1
            print('Total %d copied.' %i)
            
            save_variables_and_metagraph(sess, saver, save_to_model_dir, 'my')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load_from', type=str, help='exsting model dir.')
    parser.add_argument('--save_to', type=str, help='duplicated model dir.')
    parser.add_argument('--model', type=str, help='model name.', default='models.inception_resnet_v1')
    
    return parser.parse_args(argv)
            
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
# python ./copy_weights_from_existing_model.py --load_from='/home/lzhang/facenet/20180408-102900' --save_to='/home/lzhang/model_zoo/TensorFlow/facenet/models/my'
