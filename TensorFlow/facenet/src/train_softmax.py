from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import h5py
import math

import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import facenet
import lfw

def main(args):
    network = importlib.import_module(args.model_def)
    image_size = (args.image_size, args.image_size)
    
    subdir = datetime.strftime(datatime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    stat_file_name = os.path.join(log_dir, 'stat.h5')
    
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))
    
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    dataset = facenet.get_dataset(args.data_dir)

    if args.validation_set_split_ratio > 0.0:
        train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 
                                                   'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []    
    
    num_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    
    
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
        
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
        
    with tf.Graph().as_default():
        tf.set_random_state(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0, 'The training set should not be empty'
        
        val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)
        
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_q = tf.train.range_input_producer(range_size, num_epochs=None, shuffle=True, seed=None, capacity=32)
        index_deq_op = index_q.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
        
        num_preprocess_threads = 4
        input_q = data_flow_ops.FIFOQueue(capacity=2000000, shared_name=None, name=None,
                                         dtype=[tf.string, tf.int32, tf.int32], 
                                         shapes=[(1, ), (1, ), (1, )])
        enq_op = input_q.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enq_op')

        image_batch, label_batch = facenet.create_input_pipeline(input_q, image_size, num_preprocess_threads, batch_size_placeholder)
        
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))
        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))
        print('Building training graph')        
        
        prelogits, _ = network.inference(image_batch, args.keep_probability, phase_train=phase_train_placeholder, 
                                         bottleneck_layer_size=args.embedding_size, weight_decay=args.weight_decay)
        
        logits = slim.fully_connected(prelogits, num_classes, activation_fn=None, reuse=False, scope='Logits', 
                                     weights_initializer=slim.initializers.xavier_initializer(), 
                                     weights_regularizer=slim.l2_regularizer(args.weight_decay))
        
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)
        
        prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, num_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)        
        total_loss = tf.add_n([cross_entropy_mean] + regularization_loss, name='total_loss')

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step, 
                                                  args.learning_rate_decay_epochs * args.epoch_size, 
                                                  args.learning_rate_decay_factor, staircase=True,)
        tf.summary.scalar('learning_rate', learning_rate)
        
        train_op = facenet.train(total_loss, global_step, args.optimizer, learning_rate, 
                                args.moving_average_decay, tf.global_variables(), args.log_histograms)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        
        summary_op = tf.summary.merge_all()
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)
            
            print('Running training')
            num_steps = args.max_nrof_epochs * args.epoch_size
            num_val_samples = int(math.ceil(args.max_nrof_epochs / args.validate_every_n_epochs))
            
            stat = {
                'loss': np.zeros((num_steps,), np.float32),
                'center_loss': np.zeros((num_steps,), np.float32),
                'reg_loss': np.zeros((num_steps,), np.float32),
                'xent_loss': np.zeros((num_steps,), np.float32),
                'prelogits_norm': np.zeros((num_steps,), np.float32),
                'accuracy': np.zeros((num_steps,), np.float32),
                'val_loss': np.zeros((num_val_samples,), np.float32),
                'val_xent_loss': np.zeros((num_val_samples,), np.float32),
                'val_accuracy': np.zeros((num_val_samples,), np.float32),
                'lfw_accuracy': np.zeros((args.max_nrof_epochs,), np.float32),
                'lfw_valrate': np.zeros((args.max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_train': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((args.max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((args.max_nrof_epochs, 1000), np.float32),
              }
            
            for epoch in range(1, args.max_nrof_epochs + 1):
                step = sess.run(global_step, feed_dict=None)
                t = time.time()
                cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, 
                             image_paths_placeholder, labels_placeholder, learning_rate_placeholder, phase_train_placeholder, 
                             batch_size_placeholder, control_placeholder, global_step, total_loss, train_op, summary_op, 
                             summary_writer, regularization_losses, args.learning_rate_schedule_file, stat, cross_entropy_mean, 
                             accuracy, learning_rate, prelogits, prelogits_center_loss, args.random_rotate, args.random_crop, 
                             args.random_flip, prelogits_norm, args.prelogits_hist_max, args.use_fixed_image_standardization)
                stat['time_train'][epoch - 1] = time.time() - t
                
                if not cont:
                    break
                    
                t = time.time()
                if len(val_image_list)>0 and ((epoch-1) % args.validate_every_n_epochs == args.validate_every_n_epochs-1 or 
                                              epoch==args.max_nrof_epochs):
                    validate(args, sess, epoch, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, 
                             labels_placeholder, control_placeholder, phase_train_placeholder, batch_size_placeholder, stat, 
                             total_loss, regularization_losses, cross_entropy_mean, accuracy, args.validate_every_n_epochs, 
                             args.use_fixed_image_standardization)
                    stat['time_validate'][epoch - 1] = time.time() - t
                    
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)
                
                t = time.time()
                if args.lfw_dir:
                    evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, 
                             batch_size_placeholder, control_placeholder, embeddings, label_batch, lfw_paths, actual_issame, 
                             args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer, stat, epoch, 
                             args.lfw_distance_metric, args.lfw_subtract_mean, args.lfw_use_flipped_images, 
                             args.use_fixed_image_standardization)
                    stat['time_evaluate'][epoch - 1] = time.time() - t
                
                print('Saving statistics')
                with h5py.File(stat_file_name, 'w') as f:
                    for key, value in stat.items():
                        f.create_dataset(key, data=value)
    return model_dir
    
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step, loss, train_op, 
          summary_op, summary_writer, reg_losses, learning_rate_schedule_file, stat, cross_entropy_mean, accuracy, learning_rate, 
          prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm, prelogits_hist_max, 
          use_fixed_image_standardization):
    batch_number = 0
    
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    
    if lr <= 0:
        return False
    
    index_epoch = sess.run(index_dequeue_op)
    image_epoch = np.array(image_list)[index_epoch]
    label_epoch = np.array(label_list)[index_epoch]
    
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_path_array = np.expand_dims(np.array(image_epoch), 1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + \
        facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization    
    control_array = np.ones_like(labels_array) * control_value
    
    sess.run(enqueue_op, {image_paths_placeholder: image_path_array, labels_placeholder: labels_array, 
                          control_placeholder: control_array})
    
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True, batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, leraning_rate, prelogits_norm, 
                       accuracy, prelogits_center_loss]
        
        if batch_number % 100 == 0:
            _loss, _, _step, _reg_losses, _prelogits, _cross_entropy_mean, _lr, _prelogits_norm, _accuracy, \
            _center_loss, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
        else:
            _loss, _, _step, _reg_losses, _prelogits, _cross_entropy_mean, _lr, _prelogits_norm, _accuracy, \
            _center_loss = sess.run(tensor_list, feed_dict=feed_dict)            
            
        duration = time.time() - start_time
        stat['loss'][_step - 1] = _loss
        stat['center_loss'][_step - 1] = _center_loss
        stat['reg_loss'][_step - 1] = np.sum(_reg_losses)
        stat['xent_loss'][_step - 1] = _cross_entropy_mean
        stat['prelogits_norm'][_step - 1] = _prelogits_norm
        stat['learning_rate'][epoch - 1] = _lr
        stat['accuracy'][_step - 1] = _accuracy
        stat['prelogits_hist'][epoch - 1,:] += np.histogram(np.minimum(np.abs(_prelogits), prelogits_hist_max), 
                                                            bins=1000, range=(0.0, prelogits_hist_max))[0]
        
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, _loss, _cross_entropy_mean, np.sum(_reg_losses), 
               _accuracy, _lr, _center_loss))
        batch_number += 1
        train_time += duration
        
    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=_step)
    
    return True

def validate(args, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder, 
             control_placeholder, phase_train_placeholder, batch_size_placeholder, stat, loss, regularization_losses, 
             cross_entropy_mean, accuracy, validate_every_n_epochs, use_fixed_image_standardization):
    print('Running forward pass on validation set')
    
    num_batches = len(label_list) // args.lfw_batch_size
    num_images = num_batches * args.lfw_batch_size
    
    labels_array = np.expand_dims(np.array(label_list[:num_images]), 1)
    image_paths_array = np.expand_dims(np.array(image_list[:num_images]), 1)
    control_array = np.ones_like(labels_array, np.int32) * facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, 
                          control_placeholder: control_array})
    
    loss_array = np.zeros((num_batches, ), np.float32)
    xent_array = np.zeros((num_batches, ), np.float32)
    accuracy_array = np.zeros((num_batches, ), np.float32)
    
    start_time = time.time()
    for i in range(num_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: args.lfw_batch_size}
        _loss, _cross_entropy_mean, _accuracy = sess.run([loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)
        loss_array[i], xent_array[i], accuracy_array[i] = (_loss, _cross_entropy_mean, _accuracy)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    
    duration = time.time() - start_time
    
    val_index = (epoch - 1) // validate_every_n_epochs
    stat['val_loss'][val_index] = np.mean(loss_array)
    stat['val_xent_loss'][val_index] = np.mean(xent_array)
    stat['val_accuracy'][val_index] = np.mean(accuracy_array)

    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
          (epoch, duration, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
             control_placeholder, embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, 
             summary_writer, stat, epoch, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization):
    print('Runnning forward pass on LFW images')
    
    start_time = time.time()
    num_embeddings = len(actual_issame) * 2
    num_flips = 2 if use_flipped_images else 1
    num_images = num_embeddings * num_flips
    labels_array = np.expand_dims(np.arange(num_images, ), 1)
    image_path_array = np.expand_dims(np.repeat(np.array(image_paths), num_flips), 1)
    control_array = np.zeros_like(labels_array, np.int32)
    
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array) * facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        control_array += (labels_array % 2) * facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_path_array, labels_placeholder: labels_array, 
                          control_placeholder: control_array})
    
    assert num_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    num_batches = num_images // batch_size
    embedding_size = int(embeddings.get_shape()[1])
    embedding_array = np.zeros((num_images, embedding_size))
    pred_label_array = np.zeros((num_images, ))
    for i in range(num_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
        _embedding, _pred_label = sess.run([embeddings, labels], feed_dict=feed_dict)
        pred_label_array[_pred_label] = _pred_label
        embedding_array[_pred_label, :] = _embedding
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    
    embeddings = np.zeros((num_embeddings, embedding_size * num_flips))
    if use_flipped_images:
        embeddings[:, :embedding_size] = embedding_array[0::2, :]
        embeddings[:, embedding_size:] = embedding_array[1::2, :]
    else:
        embeddings = embedding_array
    
    assert np.array_equal(lab_array, np.arange(nrof_images))==True, \
    'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'    
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, 
                                                     distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    stat['lfw_accuracy'][epoch-1] = np.mean(accuracy)
    stat['lfw_valrate'][epoch-1] = val

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
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
    summary = tf.Summary()
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--lfw_subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))