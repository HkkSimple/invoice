import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import requests
import json
from glob import glob
from multiprocessing import Pool
from time import time
from tqdm import tqdm
import os
from pprint import pprint
import pickle as cpk
from concurrent import futures

from invoice_detect import util as ut

def request_api(url, header, params):
    response = requests.post(url, headers=header, data=json.dumps(params))
    response = response.json()
    return response

def base64_to_image(base64_str, channel=1):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    if channel == 1:
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if channel == 3:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img_path):
    with open(img_path, 'rb') as f:
        base64_bytes = base64.b64encode(f.read())
    return base64_bytes.decode()

def array_to_base64(arr):
    retval, buffer = cv2.imencode('.jpg', arr)
    arr_str = base64.b64encode(buffer)
    arr_str = arr_str.decode()
    return arr_str

def base64_to_array(base64_str):
    img_data = base64.b64decode(base64_str)
    arr = cpk.loads(img_data)
    return arr

def data_process(img):
    base64_bytes = array_to_base64(img)
    param = {"imageData": base64_bytes,
            "imageUUID": 'AI-TEST-CLIENT'}
    return param

def get_request_result(data, url, headers):
    param = data_process(data)
    rst = request_api(url, headers, param)
    return rst


def angle_recognize(base64_str, img, uuid):
    url = "http://127.0.0.1:30101/ocr/image_angle_detect"
    headers = {"Content-Type": 'application/json;charset=UTF-8'}
    param = {"imageData": base64_str,
            "imageUUID": uuid}
    response = requests.post(url, headers=headers, data=json.dumps(param))
    response = response.json()
    response = response.get('responseData', None)
    if response is None:
        angle = 0
    else:
        angle = response.get('angle', 0)
    if angle == 90 or angle == 180 or angle == 270:
        # Image 的rotate函数是逆时针，rotate_bound 是顺时针
        angle_map = {'90': 270, '180': 180, '270': 90}
        angle = angle_map[str(angle)]
        img = ut.rotate_bound(img, angle)
    return angle, img