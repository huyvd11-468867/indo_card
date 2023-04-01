"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2


def get_full_point(list_pred):
    if list_pred.count(0) == 0:
        return list_pred

    if list_pred.count(0) == 1:
        if list_pred.index(0) == 2:
            br, bl, tl = list_pred[3], list_pred[1], list_pred[0]
            center_xmin, center_ymin = (get_center(br)[0] + get_center(tl)[0]) // 2, (
                    get_center(br)[1] + get_center(tl)[1]) // 2

            x_len, y_len = get_len(bl)

            tr = [center_xmin * 2 - get_center(bl)[0] - x_len, center_ymin * 2 - get_center(bl)[1] - y_len,
                  center_xmin * 2 - get_center(bl)[0] + x_len, center_ymin * 2 - get_center(bl)[1] + y_len]

            list_pred[2] = tr

        elif list_pred.index(0) == 3:
            tr, bl, tl = list_pred[2], list_pred[1], list_pred[0]
            center_xmin, center_ymin = (get_center(tr)[0] + get_center(bl)[0]) // 2, (
                    get_center(tr)[1] + get_center(bl)[1]) // 2

            x_len, y_len = get_len(tl)

            br = [center_xmin * 2 - get_center(tl)[0] - x_len, center_ymin * 2 - get_center(tl)[1] - y_len,
                  center_xmin * 2 - get_center(tl)[0] + x_len, center_ymin * 2 - get_center(tl)[1] + y_len]

            list_pred[3] = br

        elif list_pred.index(0) == 0:
            tr, bl, br = list_pred[2], list_pred[1], list_pred[3]
            center_xmin, center_ymin = (get_center(tr)[0] + get_center(bl)[0]) // 2, (
                    get_center(tr)[1] + get_center(bl)[1]) // 2

            x_len, y_len = get_len(br)

            tl = [center_xmin * 2 - get_center(br)[0] - x_len, center_ymin * 2 - get_center(br)[1] - y_len,
                  center_xmin * 2 - get_center(br)[0] + x_len, center_ymin * 2 - get_center(br)[1] + y_len]

            list_pred[0] = tl

        elif list_pred.index(0) == 1:
            tr, br, tl = list_pred[2], list_pred[3], list_pred[0]
            center_xmin, center_ymin = (get_center(tl)[0] + get_center(br)[0]) // 2, (
                    get_center(tl)[1] + get_center(br)[1]) // 2

            x_len, y_len = get_len(tr)

            bl = [center_xmin * 2 - get_center(tr)[0] - x_len, center_ymin * 2 - get_center(tr)[1] - y_len,
                  center_xmin * 2 - get_center(tr)[0] + x_len, center_ymin * 2 - get_center(tr)[1] + y_len]

            list_pred[1] = bl

    return list_pred


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

    widthA = np.sqrt((((br[0] - bl[0]) ** 2) + (br[1] - bl[1]) ** 2))
    widthB = np.sqrt((((tr[0] - tl[0]) ** 2) + (tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [max_height - 1, 0],
                    [max_height - 1, max_width - 1],
                    [0, max_width - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_height, max_width))

    return warped


def rolling_img(img_open, dict_main):
    img_cv2 = img_open
    dem = 0
    zero_box, one_box, two_box, three_box, four_box = 0, 0, 0, 0, 0

    list_predict = []
    tl, bl, tr, br = 0, 0, 0, 0

    if dict_main:
        if dict_main.get('tl') is None:
            tl = 0
        else:
            if len(dict_main.get('tl')[0]) == 4:
                dem += 1
                tl = dict_main.get('tl')[0]

        if dict_main.get('bl') is None:
            bl = 0
        else:
            if len(dict_main.get('bl')[0]) == 4:
                dem += 1
                bl = dict_main.get('bl')[0]

        if dict_main.get('tr') is None:
            tr = 0
        else:
            if len(dict_main.get('tr')[0]) == 4:
                dem += 1
                tr = dict_main.get('tr')[0]

        if dict_main.get('br') is None:
            br = 0
        else:
            if len(dict_main.get('br')[0]) == 4:
                dem += 1
                br = dict_main.get('br')[0]

        list_predict.append(tl)
        list_predict.append(bl)
        list_predict.append(tr)
        list_predict.append(br)

        list_new_perspective = np.zeros((4, 2), dtype="float32")

        if list_predict.count(0) == 4:
            zero_box += 1
        if list_predict.count(0) == 3:
            one_box += 1
        if list_predict.count(0) == 2:
            two_box += 1
        if list_predict.count(0) == 1:
            three_box += 1
            list_predict_3point = get_full_point(list_predict)

            list_new_perspective[0] = [(list_predict_3point[0][0] + list_predict_3point[0][2]) // 2,
                                       (list_predict_3point[0][1] + list_predict_3point[0][3]) // 2]
            list_new_perspective[1] = [(list_predict_3point[1][0] + list_predict_3point[1][2]) // 2,
                                       (list_predict_3point[1][1] + list_predict_3point[1][3]) // 2]
            list_new_perspective[2] = [(list_predict_3point[2][0] + list_predict_3point[2][2]) // 2,
                                       (list_predict_3point[2][1] + list_predict_3point[2][3]) // 2]
            list_new_perspective[3] = [(list_predict_3point[3][0] + list_predict_3point[3][2]) // 2,
                                       (list_predict_3point[3][1] + list_predict_3point[3][3]) // 2]

            img_cv2 = four_point_transform(img_open, list_new_perspective)

        if list_predict.count(0) == 0:
            list_new_perspective[0] = [(list_predict[0][0] + list_predict[0][2]) // 2,
                                       (list_predict[0][1] + list_predict[0][3]) // 2]
            list_new_perspective[1] = [(list_predict[1][0] + list_predict[1][2]) // 2,
                                       (list_predict[1][1] + list_predict[1][3]) // 2]
            list_new_perspective[2] = [(list_predict[2][0] + list_predict[2][2]) // 2,
                                       (list_predict[2][1] + list_predict[2][3]) // 2]
            list_new_perspective[3] = [(list_predict[3][0] + list_predict[3][2]) // 2,
                                       (list_predict[3][1] + list_predict[3][3]) // 2]

            four_box += 1

            img_cv2 = four_point_transform(img_open, list_new_perspective)

    return img_cv2


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[0]
    rect[2] = pts[3]
    rect[1] = pts[2]
    rect[3] = pts[1]

    return rect


def get_center(box):
    return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]


def get_len(box):
    return (box[2] - box[0]) // 2, (box[3] - box[1]) // 2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    """random preprocessing for real-time data augmentation"""
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
