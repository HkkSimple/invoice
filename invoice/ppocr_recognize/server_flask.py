import os
import sys
sys.path.append('/mnt/data/rz/programe/simpleService')

from simple_service.service.default import DefaultServer
from simple_service.config.options import Options
from config.config import Config
from flask import Flask

import models
import application
#利用simpleservice框架注册了新的application,models类，所以需要在这里通过import的方式将其注册上去

app = Flask(__name__)

def main():
    opt = Options().parse()
    cfg = Config().setup(opt)

    server = DefaultServer(cfg)
    server(app)

if __name__ == '__main__':
    main()