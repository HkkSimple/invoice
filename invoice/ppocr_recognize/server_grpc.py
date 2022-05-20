import os
import sys
sys.path.append('/Users/huangkai/Documents/workstation/simpleService')

from simple_service.service.default import DefaultGRPCServer
from simple_service.config.options import Options
from config.config import Config

import models
import service
import application
#利用simpleservice框架注册了新的application,models类，所以需要在这里通过import的方式将其注册上去


def main():
    opt = Options().parse()
    cfg = Config().setup(opt)

    server = DefaultGRPCServer(cfg)
    server()

if __name__ == '__main__':
    main()