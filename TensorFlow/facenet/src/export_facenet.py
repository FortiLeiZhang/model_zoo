from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import os
import sys
import facenet

def main(args):
    model_dir = args.model_dir
    output_dir = args.output_dir
    model_version = args.model_version
    model_dir_exp = os.path.expanduser(model_dir)
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Model directory: %s' % model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(model_dir_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

            input_graph_def = sess.graph.as_graph_def()
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')

            path = os.path.join(output_dir, str(model_version))
            print('Exporting trained model to', path)
            builder = tf.saved_model.builder.SavedModelBuilder(path)

            img_str_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")

            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'images': tf.saved_model.utils.build_tensor_info(img_str_placeholder)
                },
                outputs={
                    'embeddings': tf.saved_model.utils.build_tensor_info(embeddings)
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

            legacy_init_op = tf.group(
                tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'calculate_embeddings': prediction_signature,
                })

            builder.save()
            print('Successfully exported model to ' + path)

#         with tf.gfile.GFile(args.output_file, 'wb') as f:
#             f.write(output_graph_def.SerializeToString())
#         print("%d ops in the final graph: %s" % (len(output_graph_def.node), args.output_file))

def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnetV1') or node.name.startswith('embeddings') or
                node.name.startswith('phase_train') or node.name.startswith('Bottleneck') or node.name.startswith(
                    'Logits')):
            whitelist_names.append(node.name)

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, help='Directory for metagraph (.meta) file and checkpoint (ckpt) file.')
    parser.add_argument('--output_dir', type=str, help='Output dir for the exported graphdef protobuf (.pb)')
    parser.add_argument('--model_version', type=int, help='model version', default=1)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

# python ./export_facenet.py --model_dir=/home/lzhang/model_zoo/TensorFlow/facenet/models/my --output_dir=/home/lzhang/model_zoo/TensorFlow/facenet/models/export_model --model_version=2