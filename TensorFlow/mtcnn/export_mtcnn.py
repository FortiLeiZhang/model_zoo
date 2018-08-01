import os
import tensorflow as tf

def load_model(model_path, sess):
    meta_file = os.path.join(model_path, 'model.meta')
    ckpt_file = os.path.join(model_path, 'model.ckpt')

    model = tf.train.import_meta_graph(meta_file, clear_devices=True)
    print(model_path)
    model.restore(sess, ckpt_file)

def export_mtcnn(base_path):
    export_version = 1
    
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    model_path = os.path.join(base_path, 'saved_model')
    print('Loading model from', model_path)
    load_model(model_path, sess)
    
    export_path = os.path.join(base_path, 'export_model', str(export_version))
    if os.path.exists(export_path):
        os.removedirs(export_path)
    print('Exporting model to', export_path)
    
    graph = tf.get_default_graph()
    x_pnet = graph.get_tensor_by_name('pnet/input:0')
    y_pnet1 = graph.get_tensor_by_name('pnet/p_net/conv4-2/BiasAdd:0')
    y_pnet2 = graph.get_tensor_by_name('pnet/p_net/prob1/truediv:0')
    pnet_sig = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tf.saved_model.utils.build_tensor_info(x_pnet)},
        outputs={'result1': tf.saved_model.utils.build_tensor_info(y_pnet1),
                 'result2': tf.saved_model.utils.build_tensor_info(y_pnet2)},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    x_rnet = graph.get_tensor_by_name('rnet/input:0')
    y_rnet1 = graph.get_tensor_by_name('rnet/r_net/conv5-2/BiasAdd:0')
    y_rnet2 = graph.get_tensor_by_name('rnet/r_net/prob1/Softmax:0')
    rnet_sig = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tf.saved_model.utils.build_tensor_info(x_rnet)},
        outputs={'result1': tf.saved_model.utils.build_tensor_info(y_rnet1),
                 'result2': tf.saved_model.utils.build_tensor_info(y_rnet2)},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    x_onet = graph.get_tensor_by_name('onet/input:0')
    y_onet1 = graph.get_tensor_by_name('onet/o_net/conv6-2/BiasAdd:0')
    y_onet2 = graph.get_tensor_by_name('onet/o_net/conv6-3/BiasAdd:0')
    y_onet3 = graph.get_tensor_by_name('onet/o_net/prob1/Softmax:0')
    onet_sig = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tf.saved_model.utils.build_tensor_info(x_onet)},
        outputs={'result1': tf.saved_model.utils.build_tensor_info(y_onet1),
                 'result2': tf.saved_model.utils.build_tensor_info(y_onet2),
                 'result3': tf.saved_model.utils.build_tensor_info(y_onet3)},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
        'pnet_predict':pnet_sig,
        'rnet_predict':rnet_sig,
        'onet_predict':onet_sig,
        },
        legacy_init_op=legacy_init_op,
        clear_devices=True
    )
    builder.save()
    
    print('Done exporting!')

# sudo tensorflow_model_server --port=9000 --enable_batching=true --model_config_file=/home/lzhang/model_zoo/TensorFlow/mtcnn/model.config