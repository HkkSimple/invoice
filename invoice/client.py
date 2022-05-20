import base64
import numpy as np
import cv2
import requests
import json


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

def path_image_to_base64(img_path):
    with open(img_path, 'rb') as f:
        base64_bytes = base64.b64encode(f.read())
    return base64_bytes.decode()

def image_to_base64(arr):
    retval, buffer = cv2.imencode('.jpg', arr)
    arr_str = base64.b64encode(buffer)
    arr_str = arr_str.decode()
    return arr_str


if __name__ == "__main__":
    # url = "http://47.98.153.185:64328/invoice"
    # url = "http://127.0.0.1:30500/invoice"
    url = "http://172.17.0.2:30500/invoice"
    headers = {"Content-Type": 'application/json;charset=UTF-8'}
    base64_str = path_image_to_base64('data/3.jpg')
    param = {"imageData": base64_str,
            "imageUUID": 'AILab-test'}
    result = requests.post(url, headers=headers, data=json.dumps(param)).json()
    print(result)