from lib.run import *
from easydict import EasyDict

engine_zoo = {
    'tem_seq':get_engine_tem_seq
}

def get_engine(config):
    config = EasyDict(config)
    
    agent = engine_zoo[config.engine](config)
    return agent