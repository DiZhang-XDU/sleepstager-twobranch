import torch
import torch.utils.data as data
import numpy as np
import re, os
import pickle, random
from os.path import join
from sklearn.model_selection import train_test_split
from torch.utils.data import dataset


class XY_dataset_N2One(data.Dataset):
    def __init__(self, tvt = 'train', serial_len = 5, frame_len = 3000, channel_num = 5, datasetName = 'SHHS1', 
                redir_cache = False, redir_root = False):
        super(XY_dataset_N2One, self).__init__()
        self.serial_len = serial_len
        self.frame_len = frame_len
        self.channel_num = channel_num
        if type(datasetName) is str:
            cohortName = datasetName
            datasetName = [datasetName]
        else:
            cohortName = 'Custom{:02d}'.format(len(datasetName))

        if not redir_cache:
            cache_path = './prepared_data/{:}_{:}_subj_cache.pkl'.format(tvt, cohortName)
        else:
            cache_path = join(redir_cache, '{:}_{:}_subj_cache.pkl'.format(tvt, cohortName))
        #if cache
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if not redir_root:
                self.root = cache['root']
            else:
                self.root = redir_root
            self.items_subj = cache['items_subj']
            self.items_idx = cache['items_idx']
            self.y = cache['y']
            self.len = len(self.items_subj)
            return
        #else
        
        # subject selector
        if not redir_root:
            self.root = [r'H:\data\filtered_data\subjects', r'F:\data\filtered_data_125\subjects']
        else:
            self.root = redir_root
        subj_paths = []
        subjIdx = { 'SHHS1':(0,5792),
                    'SHHS2':(5793,8443),
                    'CCSHS':(8444,8958),
                    'SOF':(8959,9411),
                    'CFS':(9412,10141),
                    'MROS1':(10142,13046),
                    'MROS2':(13047,14072),}

        for dataset in datasetName:
            for i in range(subjIdx[dataset][0], subjIdx[dataset][1] + 1):
                root = self.root[0] if i<10000 else self.root[1]
                subj_path = join(root, '{:06d}'.format(i))
                assert os.path.exists(subj_path)
                subj_paths.append (subj_path)
        # split
        train_idx, valid_idx = train_test_split(subj_paths, train_size = 0.8, random_state = 0)
        valid_idx, test_idx = train_test_split(valid_idx, train_size = 0.5, random_state = 0)

        # train_num = 50
        # train_idx, valid_idx, test_idx = person_paths[0:train_num], [person_paths[train_num]], person_paths[train_num:]
        
        # gen idx
        self.items_subj, self.items_idx, self.y = [], [], []
        tvt2paths = {'train':train_idx ,'valid':valid_idx, 'test':test_idx, 'all':subj_paths}
        subj_paths = tvt2paths[tvt]
        for subj_path in subj_paths:
            with open(join(subj_path, 'stages.pkl'), 'rb') as f:
                anno = pickle.load(f)
            frameNum = len(os.listdir(join(subj_path, 'data')))
            for i in range(frameNum - serial_len + 1):
                self.items_idx.append(i)
                self.items_subj.append(int(subj_path[-6:]))
                self.y.append(int(anno[i + serial_len//2]))

        self.len = len(self.items_subj)
        # save cache
        cache = {'root':self.root, 'items_subj': self.items_subj,'items_idx':self.items_idx, 'y': self.y}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    def __getitem__(self, index):
        # paths = self.items_subj[index]
        # paths = ['{:}\\{:06d}\\data\\{:04d}.pkl'.format(self.root, self.items_subj[index], self.items_idx[index]+n) for n in range(self.serial_len)]
        subj = self.items_subj[index]
        idx = self.items_idx[index]
        root = self.root[0] if subj<10000 else self.root[1] 
        paths = ['{:}\\{:06d}\\data\\{:04d}.pkl'.format(root, subj, idx+n) for n in range(self.serial_len)]
        X = torch.zeros(size = [self.serial_len ,self.frame_len, self.channel_num]).float()
        for i in range(self.serial_len):
            with open(paths[i], 'rb') as f_data:
                pkl = pickle.load(f_data)
            X[i] = torch.from_numpy(pkl).float()
        y = torch.tensor(self.y[index]).long()
        return X, y, subj

    def __len__(self):
        return self.len
