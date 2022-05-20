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

def rotate_bound(image, angle):
    '''
    . 旋转图片
    . @param image    opencv读取后的图像
    . @param angle    (顺)旋转角度
    '''

    (h, w) = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    # 计算图像的新边界维数
    newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
    newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    # 执行实际的旋转并返回图像
    return cv2.warpAffine(image, M, (newW, newH)) # borderValue 缺省，默认是黑色


if __name__ == "__main__":
    url = "http://127.0.0.1:30101/ocr/image_angle_detect"
    headers = {"Content-Type": 'application/json;charset=UTF-8'}

    root = '/mnt/data/rz/data/invoice/v2/images'
    img_paths = glob(os.path.join(root, 'val/*.jpg'))
    angle_map = {0:0, 90:270, 180:180, 270:90}

    multi = False
    single = True
    for imgp in tqdm(img_paths):
        if single:
            img = cv2.imread(imgp)
            result = get_request_result(img, url, headers)
            angle = result['responseData']['angle']
            angle = angle_map[angle]
            rotated_img = rotate_bound(img, angle)
            if angle != 0:
                cv2.imwrite(imgp, rotated_img)
            

        if multi:
            workers = 5
            results = []
            data = [cv2.imread(imgp) for imgp in [imgp]*workers]
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