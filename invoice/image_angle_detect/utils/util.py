import cv2
import sys
import numpy as np

def resize_image_multiple32(im, max_side_len):
    """
    resize image to a size multiple of 32 which is required by the network
    args:
        img(array): array with shape [h, w, c]
    return(tuple):
        img, (ratio_h, ratio_w)
    """
    h, w = im.shape[:2]

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    if resize_h % 32 == 0:
        resize_h = resize_h
    elif resize_h // 32 <= 1:
        resize_h = 32
    else:
        resize_h = (resize_h // 32 - 1) * 32
    if resize_w % 32 == 0:
        resize_w = resize_w
    elif resize_w // 32 <= 1:
        resize_w = 32
    else:
        resize_w = (resize_w // 32 - 1) * 32
    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            return None, (None, None)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
    except:
        print(im.shape, resize_w, resize_h)
        sys.exit(0)
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)

def normalize(im):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    im = im.astype(np.float32, copy=False)
    im = im / 255
    im[:, :, 0] -= img_mean[0]
    im[:, :, 1] -= img_mean[1]
    im[:, :, 2] -= img_mean[2]
    im[:, :, 0] /= img_std[0]
    im[:, :, 1] /= img_std[1]
    im[:, :, 2] /= img_std[2]
    channel_swap = (2, 0, 1)
    im = im.transpose(channel_swap)
    return im

def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im

def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points
    
def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect
    
def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes
