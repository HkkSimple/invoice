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

def pil_to_byte(img):
    buffered = BytesIO()
    img.save(buffered, format=img.format)
    return buffered.getvalue()

def data_process(img):
    base64_bytes = array_to_base64(img)
    param = {"imageData": base64_bytes,
            "imageUUID": 'AI-TEST-CLIENT'}
    return param

# def get_request_result(data, url, headers):
def get_request_result(data):
    url = "http://47.98.153.185:64327/crnn"
    headers = {"Content-Type": 'application/json;charset=UTF-8'}
    base64_bytes = array_to_base64(data)
    param = {"imageData": base64_bytes,
            "imageUUID": 'AI-TEST-CLIENT'}
    rst = request_api(url, headers, param)
    return rst


def multi_request(data, func, workers=1):
    results = []
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future = [executor.submit(
            func, d) for d in data]
        for ft in futures.as_completed(future):
            try:
                response_result = ft.result()
                results.append(response_result)
            except Exception as exc:
                print("generated an exception:{}".format(exc))
    return results

def crnn(img, uuid):
    url = "http://127.0.0.1:30302/crnn"
    headers = {"Content-Type": 'application/json;charset=UTF-8'}
    base64_bytes = array_to_base64(img)
    param = {"imageData": base64_bytes,
            "imageUUID": uuid}
    response = requests.post(url, headers=headers, data=json.dumps(param))
    response = response.json()
    response = response.get('responseData', None)
    score = '0'
    if response is None:
        ct = ''
    else:
        ct = response.get('content', '')
        score = response.get('scores', '0')
    return ct, score