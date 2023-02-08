import os
import time
import json
import cv2
from flask import Flask, request, jsonify

from invoice_detect.detect import detectApp
from angle_client import angle_recognize
from invoice_detect import util as ut
from crnn_client import crnn

app = Flask(__name__)

# 按天创建文件夹，并返回
def day_dir(root, create=True):
    day_str = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    day_dir = os.path.join(root, day_str)
    if (not os.path.exists(day_dir)) and create:
        os.makedirs(day_dir)
    return day_dir

# 获取指定元素得位置
@app.route('/invoice', methods=['POST'])
def get_image_content():
    result = {'responseCode': '6200', 
              'responseMSG': 'error', 
              'responseData': []}
    ele_result = []
    store_img = True
    data_store_root = '/mnt/share/img_store/invoice'
    if not os.path.exists(data_store_root):
        os.makedirs(data_store_root)
    try:
        data = json.loads(request.get_data())
        base64_image_data = data['imageData']
        uuid = data['imageUUID']
        img = ut.base64_to_image(base64_image_data, channel=3)

        # image angle recognize
        angle, img = angle_recognize(base64_image_data, img, uuid)
        if store_img:
            day_data_store_root = day_dir(data_store_root) # request image store data
            img_store_path = os.path.join(day_data_store_root, str(int(time.time()*10000000)) + '_' +uuid +'.jpg')
            cv2.imwrite(img_store_path, img)

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
        ele_result.append({'class_name': 'invoiceNo', 'location':invoiceNo_loc, 'scores':str(invoiceNo_score), 'content':invoiceNo})
        ele_result.append({'class_name':'invoiceCode', 'location':invoice_code_loc, 'scores':str(invoice_code_score), 'content':invoice_code})
        result['responseData'] = ele_result
        result['responseCode'] = '0000'
        result['responseMSG'] = 'succeed'
        return jsonify(result)
    except:
        return jsonify(
            {'responseCode': '6200', 
              'responseMSG': 'error', 
              'responseData': []})

if __name__ == "__main__":
    weights = './invoice_detect/ckpt/invoice_detect.onnx'
    DETECT_MODEL = detectApp(weights)
    app.run(debug=False,
        port='30500',
        host='0.0.0.0',
        threaded=False)