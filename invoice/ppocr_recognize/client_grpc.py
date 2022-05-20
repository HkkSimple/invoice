import grpc
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
import shutil

from service.grpc.crnn import crnn_pb2_grpc, crnn_pb2

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

def array_to_byte(img):
    return cpk.dumps(img)

def byte_to_array(img):
    return cpk.loads(img)

if __name__ == "__main__":
    ip = '127.0.0.1'
    port = '9078'
    img_path = 'test.png'
    img = cv2.imread(img_path)
    print('the image shape is:', img.shape)
    t1 = time()
    encode_img = array_to_byte(img)
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = crnn_pb2_grpc.crnnStub(channel)
    response = stub.crnn(
        crnn_pb2.crnnRequest(
            imageData = encode_img,
            imageUUID = 'AILab-test-grpc',
            information = json.dumps({'flag':'demo client'})
        )
    )
    t2 = time()
    print('use time:', t2-t1)
    print('demo server result is:', response)
    print('content:', response.responseData.content)