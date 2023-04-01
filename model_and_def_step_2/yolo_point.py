# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from PIL import Image
import cv2

from model_and_def_step_2.yolo3_point.model import yolo_eval, yolo_body, tiny_yolo_body
from model_and_def_step_2.yolo3_point.utils import letterbox_image
import os


class YOLO_box_4point(object):
    _defaults = {
        "model_path": 'weight/yolo_point_20000.h5',
        "anchors_path": 'model_and_def_step_2/model_data_point/yolo_anchors.txt',
        "classes_path": 'model_and_def_step_2/model_data_point/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.yolo_model = None
        self.input_image_shape = None
        self.colors = None
        self.iou = None
        self.score = None
        self.model_image_size = None
        self.gpu_num = None
        self.model_path = None
        self.anchors_path = None
        self.classes_path = None
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def predict_4point(self, img_open):
        img = cv2.cvtColor(img_open, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        dict_main = self.detect_image(img_pil)

        return img_open, dict_main

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), 'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                # K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        dict_out = {}
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]

            ymin, xmin, ymax, xmax = box
            ymin = int(max(0, np.floor(ymin + 0.5).astype('int32')))
            xmin = int(max(0, np.floor(xmin + 0.5).astype('int32')))
            ymax = int(min(image.size[1], np.floor(ymax + 0.5).astype('int32')))
            xmax = int(min(image.size[0], np.floor(xmax + 0.5).astype('int32')))

            # trả về dict
            list_tmp = [xmin, ymin, xmax, ymax]

            if self.class_names[c] not in dict_out.keys():
                dict_out[self.class_names[c]] = []

            dict_out[self.class_names[c]].append(list_tmp)

        return dict_out

    def close_session(self):
        self.sess.close()
