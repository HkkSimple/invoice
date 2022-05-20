from simple_service.config.config import CfgNode as CN

_C = CN()

#==============================
# service basic config
#==============================
_C.SERVICE = CN()
# service output name config
_C.SERVICE.OUTPUT = CN()
_C.SERVICE.OUTPUT.RESPONSEDATA = CN()
_C.SERVICE.OUTPUT.RESPONSEDATA.ANGLE = 'angle'
#==============================
# model config
#==============================
# model basics config
_C.MODEL = CN()
_C.MODEL.WEIGHTS = ''
_C.MODEL.GPU_ID = (-1,)