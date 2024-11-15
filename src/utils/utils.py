import functools
import json
import os
import socket
import time
from collections import OrderedDict
from datetime import datetime
from itertools import repeat
from pathlib import Path

import humanize
import numpy as np
import psutil


def replace_nested_dict_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_nested_dict_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp: # this
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def memory_summary():
    vmem = psutil.virtual_memory()
    msg = (
        f">>> Currently using {vmem.percent}% of system memory "
        f"{humanize.naturalsize(vmem.used)}/{humanize.naturalsize(vmem.available)}"
    )
    print(msg)

def dump_conn(save_dir, var, var_locators, stamp):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if 'Aij' not in var:
        raise NotImplementedError()
    res_dict = dict()
    for v in var_locators:
        res_dict[str(v)] = var['Aij'][v].item()
    json_object = json.dumps(res_dict, indent=4)
    with open(str(save_dir) + "res_" + str(stamp) + ".json", "w") as outfile:
        outfile.write(json_object)

@functools.lru_cache(maxsize=64, typed=False)
def memcache(path):
    suffix = Path(path).suffix
    print(f"loading features >>>", end=" ")
    tic = time.time()
    if suffix == ".npy":
        res = np_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix} for path {path}")
    print(f"[Total: {time.time() - tic:.1f}s] ({socket.gethostname() + ':' + str(path)})")
    return res

def np_loader(np_path, l2norm=False):
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1", allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1E-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1E-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

class DCM_init():
    def __init__(self, config_path=None):
        super(DCM_init, self).__init__()
        if config_path is None:
            raise ValueError('config path cannot be None')
        config = read_json(Path(config_path))
        if config['name'] != 'DCM' or config['arch']['type'] != 'DCM_init':
            raise ValueError('proper naming does not exist')
        else:
            config = config['arch']
        self.initial_params = {'phi':config['args']['phi']*np.exp(0.),'gamma':config['args']['gamma']*np.exp(0.),'chi':config['args']['chi']*np.exp(0.),
                               'tMTT':config['args']['tMTT']*np.exp(0.40547),'tao':config['args']['tao']*np.exp(0.9163), 'alpha':config['args']['alpha']*np.exp(0.),
                               'E0':config['args']['E0']*np.exp(0.),'V0':config['args']['V0']*np.exp(0.),'theta0':config['args']['theta0']*np.exp(0.),
                               'r0':config['args']['r0']*np.exp(0.),'epsilon':config['args']['epsilon']*np.exp(0.),'TE':config['args']['TE']*np.exp(0.),
                               'sigma':config['args']['sigma']*np.exp(0.47001),'mu':config['args']['mu']*np.exp(0.693148),'lamda':config['args']['lamda']*np.exp(-0.693147)}
    def get_init_params(self):
        return self.initial_params