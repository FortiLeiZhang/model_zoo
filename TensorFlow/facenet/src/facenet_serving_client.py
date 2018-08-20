from __future__ import print_function

import time
import os
import sys
import argparse
import numpy as np
from scipy import misc
from io import BytesIO
from PIL import Image

from grpc.beta import implementations
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

def generate_input_string(image):
    image_data_arr = []
    for i in range(image.shape[0]):
        byte_io = BytesIO()
        img = Image.fromarray(image[i, :, :, :].astype(np.uint8).squeeze())

        img.save(byte_io, 'JPEG')
        byte_io.seek(0)
        image_data = byte_io.read()
        image_data_arr.append([image_data])
    return image_data_arr

def facenet_serving(images):
    host = '127.0.0.1'
    port = 9003

    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'facenet'

    request.model_spec.signature_name = 'calculate_embeddings'

    image_data_arr = np.asarray(images)
    input_size = image_data_arr.shape[0]

    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=input_size)]
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        tensor_shape=tensor_shape_proto,
        string_val=[image_data for image_data in image_data_arr])
    request.inputs['images'].CopyFrom(tensor_proto)

    result = stub.Predict(request, 10.0)

    return [MakeNdarray(result.outputs['embeddings'])]

def main(args):
    image_dir = args.image_dir
    
    img_list = []
    for file in os.listdir(image_dir):
        with open(os.path.join(image_dir, file), 'rb') as f:
            img = f.read()
            img_list.append(img)
    images = np.stack(img_list)
    
    emb = facenet_serving(images)
    emb = np.asarray(emb).squeeze()
           
    num_images = images.shape[0]

    if args.debug:
        print('Distance matrix')
        print('    ', end='')
        for i in range(num_images):
            print('    %1d     ' % i, end='')
        print('')
        for i in range(num_images):
            print('%1d  ' % i, end='')
            for j in range(num_images):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                print('  %1.4f  ' % dist, end='')
            print('')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, help='Image directory.')
    parser.add_argument('--debug', help='Image directory.', action='store_true')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

# sudo tensorflow_model_server --port=9003 --enable_batching=true --model_config_file=/home/lzhang/model_zoo/TensorFlow/facenet/src/model.config

# python ./facenet_serving_client.py --image_dir='/home/lzhang/tmp/test_160' --debug
