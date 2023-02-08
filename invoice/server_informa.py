import os
import json
import cv2
import requests
import time
from flask import Flask, request, jsonify

from invoice_detect.detect import detectApp
from angle_client import angle_recognize
from invoice_detect import util as ut
from crnn_client import crnn

app = Flask(__name__)




#识别图片类型(发票、非发票)
def get_image_category(url, img_base64):
    header = {"Content-Type": 'application/json;charset=UTF-8'}
    params = {"imageData": img_base64,
            "imageUUID": 'AI-TEST-CLIENT',
            "information": 'test'}
    response = requests.post(url, headers=header, data=json.dumps(params))
    response = response.json()
    res_data = response.get('responseData', '')

    flag = 0
    category = 'other'
    if len(res_data)>0:
        for item in res_data:
            ct = item['content']
            if '开票日期' in ct or \
                '纳税人识别号' or \
                '发票' in ct  or \
                '税率' in ct  or \
                '单位' in ct  or \
                '校验码' in ct:
                flag += 1
        if flag >=3:
            category = 'invoice'
    return category


# 获取指定元素得位置
@app.route('/invoice/informa', methods=['POST'])
def get_image_content():
    result = {'responseCode': '6200', 
              'responseMSG': 'error', 
              'responseData': {"category":'', "data":[]}}
    ele_result = []
    store_img = True
    data_store_root = '/mnt/share/img_store/invoice_informa'
    if not os.path.exists(data_store_root):
        os.makedirs(data_store_root)
    try:
        data = json.loads(request.get_data())
        base64_image_data = data['imageData']
        uuid = data['imageUUID']

        # image category(invoice or other)
        img_category = get_image_category(URL, base64_image_data)
        img = ut.base64_to_image(base64_image_data, channel=3)
        if store_img:
            day_data_store_root = ut.day_dir(data_store_root) # request image store data
            img_store_path = os.path.join(day_data_store_root, str(int(time.time()*10000000)) + '_' +uuid +'.jpg')
            cv2.imwrite(img_store_path, img)

        if img_category == 'invoice':

            # image angle recognize
            angle, img = angle_recognize(base64_image_data, img, uuid)

            # text items detect 
            locations, scores, class_names = DETECT_MODEL.inference(img)
            
            # items recognize
            invoiceNo_score, invoice_code_score, invoiceNo_loc, invoice_code_loc, invoiceNo, invoice_code = 0, 0, '', '', '', ''
            for loc, cls, sco in zip(locations, class_names, scores):
                x1, y1, x2, y2, x3, y3, x4, y4 = loc
                cut_img = img[y1:y3, x1:x3, :]
                ct, ct_sco = crnn(cut_img, uuid)
                if cls == 'invoiceNo':
                    if float(ct_sco) > float(invoiceNo_score):
                        invoiceNo_score = ct_sco
                        invoiceNo = ct
                        invoiceNo_loc = loc.tolist()
                    continue
                if cls == 'invoiceCode':
                    if float(ct_sco) > float(invoice_code_score):
                        invoice_code_score = ct_sco
                        invoice_code = ct
                        invoice_code_loc = loc.tolist()
                    continue
                tmp = {'class_name': cls, 'location': loc.tolist(), 
                        'scores': ct_sco, 'content': ct}
                ele_result.append(tmp)
            ele_result.append({'class_name': 'invoiceNo', 
                                'location':invoiceNo_loc, 
                                'scores':str(invoiceNo_score), 
                                'content':invoiceNo})
            ele_result.append({'class_name':'invoiceCode', 
                            'location':invoice_code_loc, 
                            'scores':str(invoice_code_score), 
                            'content':invoice_code})
        result['responseData']['category'] = img_category
        result['responseData']['data'] = ele_result
        result['responseCode'] = '0000'
        result['responseMSG'] = 'succeed'
        
        
            
        return jsonify(result)
    except:
        return jsonify(
            {'responseCode': '6200', 
              'responseMSG': 'error', 
              'responseData': []})

if __name__ == "__main__":
    WEIGHTS = './invoice_detect/data/invoice_detect.onnx'
    URL = "http://47.98.153.185:64324/ocr/ppocr"
    DETECT_MODEL = detectApp(WEIGHTS)
    app.run(debug=False,
        port='30500',
        host='0.0.0.0',
        threaded=False)