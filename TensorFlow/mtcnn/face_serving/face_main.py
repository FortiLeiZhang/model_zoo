import logging
import time

import cv2
import numpy as np
from scipy import misc

import face.serving.face_detect as face

logging.basicConfig(level=logging.DEBUG)

image = misc.imread('/home/user/photo_2018-05-28_10-27-32.jpg')
result, _ = face.detect_face(image)

face_crop_margin = 10
face_size = 200

for index, bbox in enumerate(result):
    bbox_accuracy = bbox[4] * 100.0

    if bbox_accuracy < 99:
        continue

    img_size = np.asarray(image.shape)[0:2]
    top = int(np.maximum(bbox[1] - face_crop_margin / 2, 0))
    right = int(np.minimum(bbox[2] + face_crop_margin / 2, img_size[1]))
    bottom = int(np.minimum(bbox[3] + face_crop_margin / 2, img_size[0]))
    left = int(np.maximum(bbox[0] - face_crop_margin / 2, 0))

    cropped_image = image[top:bottom, left:right, :]
    cropped_image = cv2.resize(cropped_image, (face_size, face_size), interpolation=cv2.INTER_LINEAR)

    misc.imsave('/home/user/face_test_result/IMG_0262_{}.jpg'.format(str(index)), cropped_image)
