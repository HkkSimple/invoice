import os
import sys
sys.path.append('/Users/huangkai/Documents/workstation/simpleService')

import grpc
from simple_service.service import GRPC_SERVER_REGISTRY

from  .crnn import crnn_pb2, crnn_pb2_grpc

@GRPC_SERVER_REGISTRY.register()
class crnnGrpcServicer(crnn_pb2_grpc.crnnServicer):
    def __init__(self, comm, inference, cfg):
        self.cfg = cfg
        self.inference = inference
        self.comm = comm

        #output data format
        self.responseData_name = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.NAME
        self.responseCode_name = self.cfg.SERVICE.OUTPUT.RESPONSECODE
        self.responseMSG_name = self.cfg.SERVICE.OUTPUT.RESPONSEMSG
        self.Oinformation = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.INFORMATION
        self.score = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.SCORES
        self.content = self.cfg.SERVICE.OUTPUT.RESPONSEDATA.CONTENT

    def get_add_servicer_to_server(self):
        return crnn_pb2_grpc.add_crnnServicer_to_server

    def crnn(self, request, context):
        result = self.comm(self.cfg, request, self.inference).post()
        code = result.get(self.responseCode_name, '6300')
        msg = result.get(self.responseMSG_name, '内部错误')
        response_data = result.get(self.responseData_name, dict())
        content = response_data.get(self.content, '')
        score = response_data.get(self.score, '0')
        info = response_data.get(self.Oinformation, 'None')
        response = crnn_pb2.crnnResponse(responseCode=code,
                                         responseMSG=msg,
                                         responseData=crnn_pb2.crnnResponseData(
                                             content=content,
                                             score=score,
                                             information = info
                                         ))
        print('crnn servicer default function finished.')
        return response