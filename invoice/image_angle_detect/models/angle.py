import os

import tensorflow as tf
from tensorflow.python.platform import gfile

from simple_service.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class angle:
    def __init__(self, cfg):
        self.cfg = cfg

        self.gpu_id = " " if self.cfg.MODEL.GPU_ID[0] == -1 else "0" # GPU_ID is a tuple(int, ...), set gpu_id == GPU_ID[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

        self.model_file_path = self.cfg.MODEL.WEIGHTS

        self.inputImg = ''
        self.predictions = ''
        self.keep_prob = ''
        self.model = ''

    def load(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        self.model = tf.Session(config=config)
        with gfile.FastGFile(self.model_file_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.model.graph.as_default()
                tf.import_graph_def(graph_def, name='')
        self.inputImg =  self.model.graph.get_tensor_by_name('input_1:0')
        self.predictions = self.model.graph.get_tensor_by_name('predictions/Softmax:0')
        self.keep_prob = tf.placeholder(tf.float32)

    def setup(self):
        self.load()

    def set_input(self, input):
        self.batches = input.copy()

    def forward(self):
        outputs = self.model.run(self.predictions, 
                             feed_dict={self.inputImg: self.batches,
                                                       self.keep_prob: 0
                                             })
        return outputs
