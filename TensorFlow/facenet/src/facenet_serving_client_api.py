from __future__ import print_function

import time
import os
import sys
import argparse
import numpy as np
from scipy import misc
from io import BytesIO
from PIL import Image
import requests
import json
import base64
import tensorflow as tf

def call_tfserver_api(signature_name, image_data_arr):
    url = "http://127.0.0.1:9004/v1/models/facenet:predict"

    image_list = []
    for image_data in image_data_arr:
        b64 = base64.b64encode(image_data[0])
        item = ''' {"images": { "b64": "%s" } }''' % b64
        image_list.append(item)

    images = ",".join(image_list)
    data = '''
    {
      "signature_name": "%s",
      "instances": [
            %s
      ]
    }
    ''' % (signature_name, images)

    response = requests.post(url, data=data)
    json_data = json.loads(response.text)
    return json_data

def facenet_serving(image_data_arr):
    result = call_tfserver_api('calculate_embeddings', image_data_arr)
    predictions = result['predictions']
    return predictions

def main(args):
    image_dir = args.image_dir
    
    image_data_arr = []
    for file in os.listdir(image_dir):
        with open(os.path.join(image_dir, file), 'rb') as f:
            img = f.read()
#             b64 = base64.b64encode(img)
            image_data_arr.append([img])
    
    emb = facenet_serving(image_data_arr)
    emb = np.asarray(emb).squeeze()
    print(emb.shape)
           
    num_images = len(image_data_arr)

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

# sudo tensorflow_model_server --rest_api_port=9004 --model_name=facenet --model_config_file=/home/lzhang/model_zoo/TensorFlow/facenet/src/model.config

# python ./facenet_serving_client_api.py --image_dir='/home/lzhang/tmp/0000045_160' --debug
