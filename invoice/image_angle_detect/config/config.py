import os
from simple_service.config.config import Config as sscfg
from simple_service.config.config import CfgNode

class Config:

    def __init__(self):
        self.cfg = self.get_cfg()
        self.simple_service_default_cfg = sscfg().get_cfg()

    def get_cfg(self) -> CfgNode:
        from .defaults import _C
        return _C.clone()

    def _reset_new_allowed(self, node:CfgNode, flag):
        node.__dict__[node.NEW_ALLOWED] = flag
        for v in node.values():
            if isinstance(v, CfgNode):
                self._reset_new_allowed(v, flag)

    def setup(self, args):
        self._reset_new_allowed(self.simple_service_default_cfg, flag=True)
        self._reset_new_allowed(self.cfg, flag=True)
        self.cfg.merge_from_other_cfg(self.simple_service_default_cfg)

        # merge config from file
        config_file = args.config_file
        for file in config_file:
            self.cfg.merge_from_file(file)
        # merge config from file
        cmd = args.cmd
        self.cfg.merge_from_list(cmd)

        self.cfg.freeze()
        return self.cfg
