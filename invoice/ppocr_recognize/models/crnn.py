import os

from paddle import inference
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from simple_service.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class crnn:
    def __init__(self, cfg):
        self.cfg = cfg

        self.gpu_id = self.cfg.MODEL.GPU_ID[0] # GPU_ID is a tuple(int, ...), set gpu_id == GPU_ID[0]
        self.gpu_memory = self.cfg.MODEL.GPU_MEMORY
        self.enable_mkldnn = self.cfg.MODEL.ENABLE_MKLDNN
        self.cpu_math_library_num_threads = self.cfg.MODEL.CPU_MATH_LIBRARY_NUM_THREADS
        self.model_file_path = os.path.join(self.cfg.MODEL.WEIGHTS, 'model')
        self.params_file_path = os.path.join(self.cfg.MODEL.WEIGHTS, 'params')

        self.paddleocr_config = ''
        self.predictor = ''
        self.input_tensor = ''
        self.output_tensors = ''

    def load(self):
        config = inference.Config(self.model_file_path, self.params_file_path)

        # set gpu(also set gpu id) or cpu
        if -1 == self.gpu_id:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.cpu_math_library_num_threads)
            if self.enable_mkldnn:
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        else:
            config.enable_use_gpu(self.gpu_memory, self.gpu_id)
        
        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
        self.paddleocr_config = config

    def create_net(self):
        predictor = inference.create_predictor(self.paddleocr_config)
        input_names = predictor.get_input_names()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
        output_names = predictor.get_output_names()
        self.output_tensors = [predictor.get_output_handle(output_name)
                                for output_name in output_names]

        self.predictor = predictor
        self.input_tensor = input_tensor

    def setup(self):
        self.load()
        self.create_net()

    def set_input(self, input):
        self.batches = input.copy()

    def forward(self):
        self.input_tensor.copy_from_cpu(self.batches)
        self.predictor.run()
        outputs = [ot.copy_to_cpu() for ot in self.output_tensors]
        return outputs
