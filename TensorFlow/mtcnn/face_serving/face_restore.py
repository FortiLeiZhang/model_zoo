import tensorflow as tf
from scipy import misc
import face.custom_detect as face
import face.facenet as facenet
import time
import numpy as np
import cv2

img = misc.imread('/home/user/S506_006_00000042.png')

tf.reset_default_graph()

with tf.Session() as sess:
    facenet.load_model('/home/user/face_detect/face_model/')

    print("Model restored.")

    pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})
    rnet_fun = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': img})
    onet_fun = lambda img: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                    feed_dict={'onet/input:0': img})

    print('start detect')
    start = time.time() * 1000.0

    bounding_boxes, _ = face.detect_face(img, 20, pnet_fun, rnet_fun, onet_fun, [0.6, 0.7, 0.7], 0.709)

    print("time:{}".format((time.time() * 1000.0) - start))

    face_crop_margin = 10
    face_size = 200

    for index, bbox in enumerate(bounding_boxes):
        bbox_accuracy = bbox[4] * 100.0

        if bbox_accuracy < 99:
            continue

        img_size = np.asarray(img.shape)[0:2]
        top = int(np.maximum(bbox[1] - face_crop_margin / 2, 0))
        right = int(np.minimum(bbox[2] + face_crop_margin / 2, img_size[1]))
        bottom = int(np.minimum(bbox[3] + face_crop_margin / 2, img_size[0]))
        left = int(np.maximum(bbox[0] - face_crop_margin / 2, 0))

        cropped_image = img[top:bottom, left:right, :]
        cropped_image = cv2.resize(cropped_image, (face_size, face_size),
                                   interpolation=cv2.INTER_LINEAR)

        # misc.imsave('E:/IMG_0262_{}.jpg'.format(str(index)), cropped_image)

