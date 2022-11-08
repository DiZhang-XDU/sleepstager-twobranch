import torch,pickle
import os
from os.path import join
from read_data import read_data

import numpy as np
from data_config import cfgData, cfgAnno
from glob import glob
import warnings 
warnings.filterwarnings("ignore")
###
# --subjects.info
# --subjects
# ----000000
# ------000000.info
# ------stages.pkl
# ------data
# --------0000.pkl
###
def main(dataset = 'SHHS2', dir_root = r'F:\data\filtered_data_125'):
    # path init
    dir_subjects = join(dir_root, 'subjects')
    dir_subjects_info = join(dir_root, 'subjects.info')
    if not os.path.exists(dir_subjects):
        os.mkdir(dir_subjects)
    
    # match ANN files
    if cfgAnno[dataset].path is not None:
        dir_anns = glob(cfgAnno[dataset].path + '/**/*.[Xx][Mm][Ll]', recursive=True) if cfgData[dataset].path is not None else []
    # match PSG files 
    fmt = ''
    for f in cfgData[dataset].format:
        fmt += '[%s%s]'%(f.upper(),f)
    dir_psgs = glob(cfgData[dataset].path + '/**/*.[Ee][Dd][Ff]', recursive=True)
    fileIDs = [e.split('\\')[-1].split('.')[0] for e in dir_psgs]

    # global idx for counting
    i_inter = 0
    if os.path.exists(dir_subjects_info):
        with open(dir_subjects_info, 'r') as f:
            lines = f.readlines()
            assert (len(lines) - 1 == 1 + int(lines[-1][:6])) if len(lines) != 1 else True
            i_inter = len(lines) - 1
            del lines
    else:
        with open(dir_subjects_info, 'w') as f:
            f.write('idx, dataset, nsrrId, sex, age, epochs\n')
    ##########################  check it every time ! ##############################
    ###### intra-set idx (for debug resuming)
    i_intra = 0
    ################################################################################
    
    # process start
    for i in range(i_intra, len(fileIDs)):
        print('\nStart:',fileIDs[i])
        print('i_inter:{:}   i:{:}\n'.format(i_inter,i))
        # check anno exist
        if cfgAnno[dataset].path is not None:
            dir_anno = 'this_path_dose_not_exist'
            for da in dir_anns:
                if eval(cfgAnno[dataset].nid2annKey) in da:
                    dir_anno = da
                    break
            if not os.path.exists(dir_anno):
                print('【ERROR!】i:{:}================================\n'.format(i))
                continue
        else:
            dir_anno = None
        # read data
        X,y,exclude = read_data(dir_psgs[i], dir_anno, dataset=dataset, sampling_rate=125, format=cfgAnno[dataset].format)
        if exclude:
            with open(join(dir_root,'exclude.info'), 'a') as f:
                f.write('%06d, %s, %s\n'%(i_inter, dataset, dir_anno))
        
        # save dir
        dir_subj = join(dir_subjects, '{:06d}'.format(i_inter))
        dir_subj_stages = join(dir_subj, 'stages.pkl')
        dir_subj_data = join(dir_subj, 'data')
        os.mkdir(dir_subj)
        os.mkdir(dir_subj_data)
        # save y
        if y is not None:
            y = torch.from_numpy(y).float()
            with open(dir_subj_stages, 'wb') as f:
                pickle.dump(y, f)
        # save X
        for j in range(X.shape[0]):
            dir_pkl = os.path.join(dir_subj_data, '{:04d}.pkl'.format(j))
            with open(dir_pkl, 'wb') as f:
                pickle.dump(X[j],f)
        # save Info
        with open(dir_subjects_info, 'a') as f:
            f.write('{:06d}, {:}, {:}, {:d}\n'.format(i_inter, dataset, fileIDs[i], len(y)))

        i_inter += 1


if __name__ == '__main__':
    main('SHHS1')
    main('SHHS2')
    main('CCSHS')
    main('SOF')
    main('CFS')
    main('MROS1')
    main('MROS2')
    
    print('done')