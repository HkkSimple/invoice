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
from utils.util import draw_text_det_res

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


if __name__ == "__main__":
    url = "http://127.0.0.1:30101/ocr/image_angle_detect"
    headers = {"Content-Type": 'application/json;charset=UTF-8'}

    multi = False
    single = True
    img_path = './img/angle_0.png'
    if single:
        img = cv2.imread(img_path)
        print('the image shape is:', img.shape)
        t1 = time()
        result = get_request_result(img, url, headers)
        t2 = time()
        print('use time:', t2-t1)
        print('demo server result is:', result)

    if multi:
        workers = 5
        results = []
        data = [cv2.imread(imgp) for imgp in [img_path, './img/angle_90.jpg']*workers]
        t1 = time()
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future = [executor.submit(
                get_request_result, d, url, headers) for d in data]
            for ft in futures.as_completed(future):
                try:
                    response_result = ft.result()
                    results.append(response_result)
                except Exception as exc:
                    print("generated an exception:{}".format(exc))
        t2 = time()
        print('use time:', t2-t1)
        print('demo server result is:', results)