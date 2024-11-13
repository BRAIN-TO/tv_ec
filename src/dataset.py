import torch
import torch.nn as nn
import os
import copy
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from random import shuffle
from utils.utils import *
from tqdm import tqdm

class BOLDdataset(Dataset):

    def __init__(self, args, root_dir='', filename='input_fmri.pt', mod=False):

        self.root_dir = root_dir
        dtype = torch.FloatTensor
        data = torch.load(root_dir + filename)['data']
        if mod == False:
            data_mod = data * 0
            data_mod[0,:] = 0
            data_mod[1,:] = 0
        val = torch.load(root_dir + filename)['target']
        assert args.n_stim == data.shape[0]
        assert args.n_reg == val.shape[0]
        uy = 0
        u_curr_1_list = []
        u_mod_1_list = []
        val_1_list = []
        lk_list = []
        self.val_shape = val.shape[1]
        num = args.window_size*int(1/args.dt)
        self.num = num
        for lk in tqdm(range(num,val.shape[1],int(1/args.dt)), desc = 'Preparing Dataset'):
            u_curr_1_list += [Variable(torch.from_numpy(data[:,uy:lk+1].reshape(args.n_stim,lk+1-uy)).type(dtype), requires_grad=args.req_grad)]
            u_mod_1_list += [Variable(torch.from_numpy(data_mod[:,uy:lk+1].reshape(args.n_stim,lk+1-uy)).type(dtype), requires_grad=args.req_grad)]
            val_1_list += [val[:,uy:lk+1].reshape(args.n_reg,lk+1-uy)]
            lk_list += [lk]

            uy = uy + int(1/args.dt)
        
        self.u_curr_1_list = u_curr_1_list
        self.u_mod_1_list = u_mod_1_list
        self.val_1_list = val_1_list
        self.lk_list = lk_list
        self.val = val

    def get_val(self):
        return self.val

    def __len__(self):
        return len(self.u_curr_1_list)

    def __getitem__(self, idx):

        record = {'u_dir': self.u_curr_1_list[idx],
                  'u_mod': self.u_mod_1_list[idx],
                  'val': self.val_1_list[idx],
                  'lk': self.lk_list[idx],
                  'num': self.num,
                  'val_shape': self.val_shape
                  }
        return record