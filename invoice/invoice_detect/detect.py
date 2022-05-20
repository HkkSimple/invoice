import os
import math
import onnxruntime

import numpy as np

from . import util as ut


"""
    use register to define a application
"""
class detectApp:
    def __init__(self, weights):
        self.max_size = 1024
        self.weights = weights
        # yolo model param
        self.classes = ('invoiceNo', 'invoiceDate', 
                        'buyerName', 'totalAmount', 
                        'priceAndTax', 'verification', 
                        'invoiceCode')
        self.filter_classes = None
        self.conf_thres = 0.45
        self.iou_thres = 0.45
        self.agnostic_nms = False

        self.model = None
        self.org_image_shape = None
        self.stride = 64

        self.init_model()

    def check(self, img_size, s=32):
        s = int(s)
        new_size = max(math.ceil(img_size / s) * s, 0)  # ceil gs-multiple
        if new_size != img_size:
            print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
        return new_size

    # create and load the model
    def init_model(self):
        self.model = onnxruntime.InferenceSession(self.weights, None)
        self.outputs_name = self.model.get_outputs()[0].name
        self.inputs_names = self.model.get_inputs()[0].name
        self.max_size = self.check(self.max_size, self.stride) # check some input param ruly

    def decode_data(self, img, channel=3):
        try:
            if type(img) == str:
                img = ut.base64_to_image(img, channel)
            self.org_image_shape = img.shape # store original image shape
            return img
        except:
            img = np.zeros((self.max_size, self.max_size, channel))
            self.org_image_shape = img.shape # store original image shape
            return img
        
    def decode_predict_data(self, data):
        pred, new_img, org_img = data # scales: ratio_h, ratio_w
        pred = ut.non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms)

        locations, confs, classes = [], [], []
        if len(pred) > 0:
            det = pred[0]
            if len(det):
                det[:, :4] = ut.scale_coords(new_img.shape[1:], det[:, :4], org_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    loc = [x1, y1, x2, y1, x2, y2, x1, y2]
                    cls = self.classes[int(cls)]
                    locations.append(loc)
                    confs.append(round(float(conf), 4))
                    classes.append(cls)
        locations = np.array(locations, dtype='int')
        return (locations, confs, classes)
    
            
    def preprocessing(self, img):
        stride = self.stride
        max_size = self.max_size

        img_org = self.decode_data(img)
        img = ut.letterbox(img_org, new_shape=max_size, stride=stride)[0]
        img = np.float32(img)
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR TO RGB, change the channel order
        img = np.ascontiguousarray(img)
        img = img / 255
        if len(img.shape) == 3:
            img = img[None]
        return (img, img_org)

    def inference(self, base64_img):
        try:
            if self.model is None:
                self.init_model()
            img, img_org = self.preprocessing(base64_img)
            pred = self.model.run([self.outputs_name], 
                                {self.inputs_names: img})[0]
            location, confs, classes = self.decode_predict_data((pred, img[0], img_org))
            return location, confs, classes
        except:
            return [], [], []

# 获取指定字段得图片
def get_item_img(img, weights):
    MODEL = detectApp(weights)
    cut_imgs = []
    img_names = []
    locations, scores, class_names = MODEL.inference(img)
    for loc, cls, sco in zip(locations, class_names, scores):
        x1, y1, x2, y2, x3, y3, x4, y4 = loc
        cut_img = img[y1:y3, x1:x3, :]
        cut_imgs.append(cut_img)
        imgn = str(x1) + "_" + str(y1) + "_" + cls + '.jpg'
        img_names.append(imgn)
    return cut_imgs, img_names


# if __name__ == "__main__":
#     MODEL = detectApp()
#     img_paths = glob('/mnt/data/rz/data/idCard/v4/cornered/person/split/0/*.jpg')
#     imgp = img_paths[1]
#     imgn = os.path.basename(imgp)
#     img = cv2.imread(imgp)
#     cv2.imwrite('data/'+imgn, img)
#     cornered_img = get_item_img(img)