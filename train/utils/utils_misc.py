import os
from termcolor import colored
from datetime import datetime
import socket
import numpy as np
from tqdm import tqdm
import shutil, errno
from pathlib import Path
import statistics
import torch
import logging
import argparse

def basic_logger(name='basic_logger'):
    logger = logging.getLogger(name)
    return logger

# Training
def red(text):
    return colored(text, 'yellow', 'on_red')

def print_red(text):
    print(red(text))

# Data
def white_blue(text):
    coloredd = colored(text, 'white', 'on_blue')
    return coloredd

def white_magenta(text):
    coloredd = colored(text, 'white', 'on_magenta')
    return coloredd

def blue_text(text):
    coloredd = colored(text, 'blue')
    return coloredd

def print_white_blue(text):
    print(white_blue(text))

def green(text):
    coloredd = colored(text, 'blue', 'on_green')
    return coloredd

def print_green(text):
    print(green(text))

def yellow(text):
    coloredd = colored(text, 'blue', 'on_yellow')
    return coloredd

# Model
def magenta(text):
    coloredd = colored(text, 'white', 'on_magenta')
    return coloredd

def print_magenta(text):
    print(magenta(text))

def copy_file(origin_dest):
    os.system('cp %s %s/'%(origin_dest[0], origin_dest[1]))

def copy_py_files(root_path, dest_path, exclude_paths=[]):
    from multiprocessing import Pool
    origin_path_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if ((file.endswith(".py") or file.endswith(".yaml"))) and (not file.endswith(".pyc")):
                origin_path = os.path.join(root, file)
                # print(os.path.join(root, file))
                exclude_flag = False
                for exclude_path in exclude_paths:
                    if exclude_path != '' and exclude_path in origin_path:
                        exclude_flag = True
                        break
                else:
                    origin_path_list.append([origin_path, dest_path])
                    # os.system('cp %s %s/'%(origin_path, dest_path))
                    # print('Copied ' + origin_path)

    with Pool(processes=12, initializer=np.random.seed(123456)) as pool:
        for _ in list(tqdm(pool.imap_unordered(copy_file, origin_path_list), total=len(origin_path_list))):
            pass

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expectedl; got: %s'%v)


def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S")
    return d1

def tryPort(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.bind(("0.0.0.0", port))
        result = True
    except:
        print("Port is in use")
    sock.close()
    return result

def nextPort(port):
    assert isinstance(port, int), 'port number should be int! e.g. 6006'
    while not tryPort(port):
        port += 1
    return port

def checkEqual1(iterator):
    """Check if elements in a list are all equal"""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def get_key(dict_in, key_in, if_bool=False, default_bool=False, default=None):
    if key_in in dict_in:
        return dict_in[key_in]
    else:
        if if_bool:
            return default_bool
        else:
            return default

def randomize():
    np.random.seed()
    
def dict_get_with_key_list(x_dict, key_list):
    return_list = []
    for key in key_list:
        assert key in x_dict, '[dict_get_with_key_list] Key %s not found in dict!'%key
        return_list.append(x_dict[key])
    if len(return_list) == 1:
        return return_list[0]
    return return_list

def flatten_list(list_of_lists):
    # flatten = lambda t: [item for sublist in list_of_lists for item in sublist]
    flat_list = [item for sublist in list_of_lists for item in sublist]
    return flat_list

from datetime import datetime

def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S-")
    return d1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, meter_name='N/A'):
        self.meter_name = meter_name
        self.val = None
        self.avg = None
        self.median = None
        self.sum = None
        self.count = None
        self.all_list = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.median = 0
        self.sum = 0
        self.count = 0
        self.all_list = []

    def update(self, val, n=1):
        # if self.meter_name == 'inv_depth_median_error_meter':
        #     print('>>>>>>> val: ', val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.all_list.append(val)
        # if self.meter_name == 'inv_depth_median_error_meter':
        #     print('-------', self.all_list)

    def get_median(self):
        return statistics.median(self.all_list)

class ListMeter(object):

    def __init__(self, meter_name='N/A'):
        self.meter_name = meter_name
        # self.val = None
        # self.avg = None
        # self.median = None
        # self.sum = None
        self.count = None
        self.all_list = []
        self.reset()

    def reset(self):
        # self.val = 0
        # self.avg = 0
        # self.median = 0
        # self.sum = 0
        self.count = 0
        self.all_list = []

    def update(self, val, n=1):
        # if self.meter_name == 'inv_depth_median_error_meter':
        #     print('>>>>>>> val: ', val)
        # self.val = val
        # self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count
        self.all_list.append(val)
        # if self.meter_name == 'inv_depth_median_error_meter':
        #     print('-------', self.all_list)

    def concat(self):
        return torch.cat(self.all_list)


# def get_time_meters():
#     time_meters = {}
#     time_meters['data_to_gpu'] = AverageMeter()
#     time_meters['forward'] = AverageMeter()
#     time_meters['loss'] = AverageMeter()
#     time_meters['backward'] = AverageMeter()    
#     time_meters['ts'] = 0

#     return time_meters    

def only1true(l):
    true_found = False
    for v in l:
        if v:
            # a True was found!
            if true_found:
                # found too many True's
                return False 
            else:
                # found the first True
                true_found = True
    # found zero or one True value
    return true_found

def nonetrue(l):
    return not any(l)