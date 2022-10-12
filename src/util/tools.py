import ast
from datetime import datetime
import numpy as np
import argparse
import os
import importlib
import random
import tensorflow as tf
import segmentation_models as sm


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', 'Y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', 'N', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2str_or_none(v):
    if v.lower() == "none":
        return None
    return v


def str2dict(v):
    return ast.literal_eval(v)


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(',')]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(',')]



def datestr():
    pacific = timezone('US/Pacific')
    now = datetime.now(pacific)
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)

def getClass(module_name, class_name):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_



def set_seed(seed):
    tf.random.set_seed(seed)
    #np.random.seed(seed)
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state
    
def write2file(arguments_dict, filename):
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)

    with open(filename, 'w') as file:
        file.write("{\n")
        for key, value in arguments_dict:
            file.write('%s: %s\n' % (key, value))


def save_model(model, path):
    model.save_weights(path)

def save_dataset(dataset, path):
    tf.data.experimental.save(dataset, path)


def load_model(model, path):
    model.load_weights(path)
    return model