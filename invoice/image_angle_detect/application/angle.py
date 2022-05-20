import cv2
import json
import logging
import traceback
import numpy as np
from copy import deepcopy

from simple_service.models import build_model
from simple_service.utils import images as uim
from simple_service.application.base import BaseApplication
from simple_service.application import APPLICATION_REGISTRY

from utils import util as ut

"""
    use register to define a application
"""
@APPLICATION_REGISTRY.register()
class angleApp(BaseApplication):
    def __init__(self, cfg):
        self.cfg = cfg

        self.input_param_img_name = self.cfg.SERVICE.INPUT.IMAGE
        self.input_param_information_name = self.cfg.SERVICE.INPUT.INFO
        self.img_channel = self.cfg.MODEL.IMAGE.CHANNEL
        self.Oinfo = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.INFORMATION
        self.angle = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.ANGLE

        # angle model params
        self.logger = logging.getLogger(self.cfg.LOG.LOG_NAME)
        self.ROTATE = [0, 90, 180, 270]

        self.model = None

    # create and load the model
    def init_model(self):
        self.model = build_model(self.cfg)
        self.model.setup()

    def decode_data(self, data, channel):
        img = data[self.input_param_img_name]
        try:
            if type(img) == str:
                img = uim.base64_to_image(img, channel)
            elif type(img) == bytes:
                img = uim.bytes_to_image(img)
            else:
                raise RuntimeError('str and bytes can support. the {} type is not support'.format(type(img)))
            infor = data.get(self.input_param_information_name, {'flag':'None'})
            self.org_image_shape = img.shape # store original image shape
            return img, infor
        except:
            img = np.zeros((244, 224, channel))
            infor = data.get(self.input_param_information_name, {'flag':'None'})
            self.org_image_shape = img.shape # store original image shape
            log_info = {"inputData":None, "outputData":None, "error":traceback.format_exc()}
            self.logger.info(json.dumps(log_info))
            return img, infor
        
    def decode_predict_data(self, data):
        pred, info = data 
        index = np.argmax(pred)
        angle = self.ROTATE[index]
        return angle, info
    
    def format_output_data(self, data):
        angle, info = data
        frt_data = {self.angle: angle,
                    self.Oinfo:info}
        return frt_data
            
    def preprocessing(self, data):
        channel = self.img_channel

        img, infor = self.decode_data(data, channel)
        h, w, = img.shape[:2]
        adjust = True
        if adjust:
            thesh = 0.05
            xmin,ymin,xmax,ymax = int(thesh*w),int(thesh*h),w-int(thesh*w),h-int(thesh*h)
            img = img[ymin:ymax,xmin:xmax]##剪切图片边缘
        img = cv2.resize(img,(224,224))
        img = img[..., ::-1].astype(np.float32)
            
        img[..., 0] -= 103.939
        img[..., 1] -= 116.779
        img[..., 2] -= 123.68
        return img, infor

    def inference(self, data):
        try:
            if self.model is None:
                self.init_model()
            length = len(data)
            infors = [d[-1] for d in data]
            batch_images = []
            for d in data:
                img = d[0]
                batch_images.append(img)
            batch_images = np.array(batch_images)

            self.model.set_input(batch_images)
            result = self.model.forward()
            return list(zip(result, infors))
        except:
            log_info = {"inputData":None, "outputData":None, "error":traceback.format_exc()}
            self.logger.info(log_info)
            return [None]*length