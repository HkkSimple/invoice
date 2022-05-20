from flask import Flask, request, jsonify
from detect import detectApp
import json

app = Flask(__name__)

# 获取指定元素得位置
@app.route('/invoice', methods=['POST'])
def get_image_content():
    result = {'responseCode': '6200', 
              'responseMSG': 'error', 
              'responseData': {}}
    ele_result = []
    try:
        data = json.loads(request.get_data())
        base64_image_data = data['imageData']
        uuid = data['imageUUID']

        locations, scores, class_names = DETECT_MODEL.inference(base64_image_data)
        for loc, cls, sco in zip(locations, class_names, scores):
            tmp = {'class_name': cls, 'location': loc.tolist(), 'scores': sco}
            ele_result.append(tmp)
        result['responseData'] = ele_result
        
        return jsonify(result)
    except:
        return jsonify(
            {'responseCode': '6200', 
              'responseMSG': 'error', 
              'responseData': {}})

if __name__ == "__main__":
    weights = './data/invoice_detect.onnx'
    DETECT_MODEL = detectApp(weights)
    app.run(debug=False,
        port='30401',
        host='0.0.0.0',
        threaded=False)