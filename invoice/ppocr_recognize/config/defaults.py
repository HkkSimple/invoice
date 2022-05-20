from simple_service.config.config import CfgNode as CN

_C = CN()

#==============================
# service basic config
#==============================
_C.SERVICE = CN()
# service output name config
_C.SERVICE.OUTPUT = CN()
_C.SERVICE.OUTPUT.RESPONSEDATA = CN()
_C.SERVICE.OUTPUT.RESPONSEDATA.SCORES = 'scores'
_C.SERVICE.OUTPUT.RESPONSEDATA.CONTENT = 'content'
#==============================
# model config
#==============================
# model basics config
_C.MODEL = CN()
_C.MODEL.WEIGHTS = ''
_C.MODEL.GPU_ID = (-1,)
_C.MODEL.IR_OPTIM = True
_C.MODEL.USE_TENSORRT = False
_C.MODEL.ENABLE_MKLDNN = False
_C.MODEL.CPU_MATH_LIBRARY_NUM_THREADS = 4
_C.MODEL.MAX_CANDIDATES = 1000
_C.MODEL.GPU_MEMORY = 8000
_C.MODEL.MAX_SIZE = 960
_C.MODEL.CHAR_TYPE = 'ch'
_C.MODEL.CHARACTER_DICT_PATH = ''
_C.MODEL.USE_SPACE_CHAR = False
