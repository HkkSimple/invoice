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
from utils.crnn_postprocess import CTCLabelDecode
"""
    use register to define a application
"""
@APPLICATION_REGISTRY.register()
class crnnApp(BaseApplication):
    def __init__(self, cfg):
        self.cfg = cfg

        self.input_param_img_name = self.cfg.SERVICE.INPUT.IMAGE
        self.input_param_information_name = self.cfg.SERVICE.INPUT.INFO
        self.img_channel = self.cfg.MODEL.IMAGE.CHANNEL
        self.max_size = self.cfg.MODEL.IMAGE.MAX_SIZE # the image width
        self.Oinfo = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.INFORMATION
        self.score = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.SCORES
        self.content = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.CONTENT
        # self.server_comm_mode = 'grpc' if self.cfg.SERVICE.GRPC_SERVICER != '' else 'flask' # service communication method used by the project

        # crnn model params
        self.character_dict_path = self.cfg.MODEL.CHARACTER_DICT_PATH
        self.character_type = self.cfg.MODEL.CHAR_TYPE
        self.use_space_char = self.cfg.MODEL.USE_SPACE_CHAR

        self.crnn_postprocess = CTCLabelDecode(self.character_dict_path, self.character_type, self.use_space_char)
        self.logger = logging.getLogger(self.cfg.LOG.LOG_NAME)
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
            img = np.zeros((32, 224, channel))
            infor = data.get(self.input_param_information_name, {'flag':'None'})
            self.org_image_shape = img.shape # store original image shape
            log_info = {"inputData":None, "outputData":None, "error":traceback.format_exc()}
            self.logger.info(json.dumps(log_info))
            return img, infor
        
    def decode_predict_data(self, data):
        pred, info = data 
        text, score = self.crnn_postprocess(pred)
        return text, score, info
    
    def format_output_data(self, data):
        text, score, info = data
        frt_data = {self.content: text,
                    self.score: str(score),
                    self.Oinfo:info}
        return frt_data
            
    def preprocessing(self, data):
        channel = self.img_channel
        max_size = self.max_size

        decoded, infor = self.decode_data(data, channel)
        h, w, = decoded.shape[:2]
        ratio = w * 1.0 / h
        imgW = int(32 * ratio) # resize img to height:32, width <= max_size
        if imgW > max_size:
            imgW = max_size
        resized_img = cv2.resize(decoded, (imgW, 32))
        resized_img = resized_img.astype("float32")
        resized_img = resized_img.transpose((2, 0, 1)) / 255
        resized_img -= 0.5
        resized_img /= 0.5
        return resized_img, infor

    def inference(self, data):
        try:
            if self.model is None:
                self.init_model()
            channel = self.img_channel
            length = len(data)
            infors = [d[-1] for d in data]
            if length == 1: # single image request
                img = data[0][0]
                template_images = img.copy()[np.newaxis, :]
            else:
                max_w = max([d[0].shape[2] for d in data])
                T = np.zeros((channel, 32, max_w), dtype=np.float32)
                template_images = []
                for d in data:
                    template = deepcopy(T)
                    img = d[0]
                    h, w = img.shape[1:3]
                    template[:, :, 0:w] = img
                    template_images.append(template)
                template_images = np.array(template_images)

            self.model.set_input(template_images)
            result = self.model.forward()[0]
            return list(zip(result, infors))
        except:
            log_info = {"inputData":None, "outputData":None, "error":traceback.format_exc()}
            self.logger.info(json.dumps(log_info))
            return [None]*length