import os

import tensorflow as tf


def load_model(path, sess):
    """
    Load the learned model from a specific path.

    :param model_path: str
        The path to the model to load.
    """
    if not isinstance(path, str):
        raise TypeError("type of 'path' must be str.")

    par_dir = os.path.split(path)[0]
    meta_file = '{}.meta'.format(path)
    ckpt_file = '{}/checkpoint'.format(par_dir)

    if not os.path.exists(meta_file):
        raise Exception('The meta file is not exist from: {}'.format(par_dir))

    if not os.path.exists(ckpt_file):
        raise Exception('The checkpoint file is not exist from: {}'.format(par_dir))

    model = tf.train.import_meta_graph(meta_file, clear_devices=True)
    model.restore(sess, tf.train.latest_checkpoint(par_dir))
    return model


def export2(model_path):
    if not isinstance(model_path, str):
        raise TypeError("type of 'model_path' must be str.")

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    export_path_base = './model'
    export_version = 1

    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(export_version)))

    if os.path.exists(export_path):
        os.removedirs(export_path)

    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    load_model(model_path, sess)

    graph = tf.get_default_graph()

    print("get tensors from graph")
    x_pnet = graph.get_tensor_by_name('pnet/input:0')
    y_pnet1 = graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
    y_pnet2 = graph.get_tensor_by_name('pnet/prob1:0')

    x_rnet = graph.get_tensor_by_name('rnet/input:0')
    y_rnet1 = graph.get_tensor_by_name('rnet/conv5-2/conv5-2:0')
    y_rnet2 = graph.get_tensor_by_name('rnet/prob1:0')

    x_onet = graph.get_tensor_by_name('onet/input:0')
    y_onet1 = graph.get_tensor_by_name('onet/conv6-2/conv6-2:0')
    y_onet2 = graph.get_tensor_by_name('onet/conv6-3/conv6-3:0')
    y_onet3 = graph.get_tensor_by_name('onet/prob1:0')

    pnet_signature = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tf.saved_model.utils.build_tensor_info(x_pnet)},
        outputs={'result1': tf.saved_model.utils.build_tensor_info(y_pnet1),
                 'result2': tf.saved_model.utils.build_tensor_info(y_pnet2)},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    rnet_signature = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tf.saved_model.utils.build_tensor_info(x_rnet)},
        outputs={'result1': tf.saved_model.utils.build_tensor_info(y_rnet1),
                 'result2': tf.saved_model.utils.build_tensor_info(y_rnet2)},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    onet_signature = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tf.saved_model.utils.build_tensor_info(x_onet)},
        outputs={'result1': tf.saved_model.utils.build_tensor_info(y_onet1),
                 'result2': tf.saved_model.utils.build_tensor_info(y_onet2),
                 'result3': tf.saved_model.utils.build_tensor_info(y_onet3)},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'pnet_predict': pnet_signature,
            'rnet_predict': rnet_signature,
            'onet_predict': onet_signature,
        },
        legacy_init_op=legacy_init_op,
        clear_devices=True
    )

    builder.save()

    print('Done exporting!')


if __name__ == '__main__':
    export2('/home/user/face_detect/face_model/')
