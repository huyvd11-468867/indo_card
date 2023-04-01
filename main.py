import base64
import glob, os
import io
import json

import cv2
from PIL import Image

from model_and_def_step_1.yolo3.utils import cropping, make_padding
from model_and_def_step_1.yolo_card import YOLO_box_card

from model_and_def_step_2.yolo3_point.utils import rolling_img
from model_and_def_step_2.yolo_point import YOLO_box_4point

from model_and_def_step_3.yolo_attribute import YOLO_atb

from model_crnn_step4_NIK.NIK_predict import NIK_PredictCRNN
from model_crnn_step4_date.date_predict import date_PredictCRNN
from model_crnn_step4_atb.atb_predict import atb_PredictCRNN


class ExtractIndoCard:
    def __init__(self):
        self.model_step_one = YOLO_box_card()
        self.model_step_two = YOLO_box_4point()
        self.model_step_three = YOLO_atb()
        self.model_atb = atb_PredictCRNN()
        self.model_date = date_PredictCRNN()
        self.model_NIK = NIK_PredictCRNN()

    def predict(self, img_pred):
        list_img_out, dict_atb, dict_main = [], {}, {}
        img_cord_list = self.model_step_one.predict_card(img_pred)
        if img_cord_list:
            for img_cord in img_cord_list:
                crop_img = cropping(img_pred, img_cord)
                add_black = make_padding(crop_img)
                img_cv2, img_4p_cord = self.model_step_two.predict_4point(add_black)

                if img_4p_cord is not None:
                    img_four_point = rolling_img(img_cv2, img_4p_cord)
                    list_img_out.append(img_four_point)
                else:
                    return dict_atb
        else:
            return dict_atb

        img_out = list_img_out[0]
        dict_atb = self.model_step_three.predict_atb(img_out)

        for atb_key in dict_atb.keys():
            for i, img_cord in enumerate(dict_atb[atb_key]):
                img_pred_step4 = img_out[img_cord[1]: img_cord[3], img_cord[0]: img_cord[2]]
                img_pred_step4 = cv2.cvtColor(img_pred_step4, cv2.COLOR_BGR2GRAY)
                pil_image = Image.fromarray(img_pred_step4)

                if atb_key == 'NIK':
                    text_pred = self.model_NIK.predict(pil_image)
                    dict_atb[atb_key][i] = text_pred
                    continue

                if atb_key == 'Tgl Lahir':
                    text_pred = self.model_date.predict(pil_image)
                    dict_atb[atb_key][i] = text_pred
                    continue

                text_pred = self.model_atb.predict(pil_image)
                dict_atb[atb_key][i] = text_pred

        for key in dict_atb.keys():
            values = ' '.join(str(val) for val in dict_atb[key])
            dict_atb[key] = values

        with io.BytesIO() as buf:
            img_pred.save(buf, 'jpeg')
            img_pred = buf.getvalue()

        str_base64 = str(base64.b64encode(img_pred))
        print(type(str_base64))
        dict_main['info'] = dict_atb
        dict_main['img'] = str_base64

        print(dict_main)
        return dict_main

#
# img_path = '/home/huyvd/Documents/works/pytorch_tutorial/tap_code/indo_card_code/test_end'
# arr_img = glob.glob(os.path.join(img_path, '*'))
#
# model = ExtractIndoCard()
#
# for img in arr_img:
#     name = os.path.basename(img)
#     img_open = Image.open(img)
#     dict_end = model.predict(img_open)
#     # print(dict_end)
#
#     print(json.dumps(dict_end))
#     cv2.waitKey()
