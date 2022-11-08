import warnings
import torch,os
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import dataset
from tqdm import tqdm
import numpy as np
from tools.data_loader_cohort_subj import XY_dataset_N2One as myDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from torch.nn import DataParallel as DP
from tools.metrics import acc_stage


os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
import warnings
warnings.filterwarnings("ignore")##################################
# !!!!!!!!!!!!!!!!! DataLoader 
import multiprocessing
multiprocessing.set_start_method('spawn', True)





def tester(cfg, alpha=.5):
    saveObj = torch.load(cfg.best_ckp)
    model = saveObj['net'].cuda()
    # model = myNet().cuda()
    # model.load_state_dict(model_.state_dict())

    
    if type(model) == DP:
        model = model.module
    if cfg.eval_parallel:
        model = DP(model)

    try:
        setName = cfg.test_dataset
    except:
        setName = cfg.dataset
    testSet = myDataset(tvt = cfg.tvt, frame_len = 30*cfg.freq, datasetName=setName, 
                        redir_root = cfg.redir_root, redir_cache = cfg.redir_cache)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = cfg.BATCH_SIZE*2, 
                                    shuffle = False, num_workers = cfg.eval_thread, drop_last = False)

    model.eval()
    ###

    subjs_target = {}
    subjs_pred = {}
    ###

    with torch.no_grad():
        tq = tqdm(testLoader, desc= 'Test', ncols=80, ascii=True)
        for i, (data, target, subjs) in enumerate(tq):
            inputs = Variable(data.cuda())
            targets = Variable(target.cuda())
            x1, x2, loss1, loss2 = model(inputs, targets.view([-1]).long())
            pred = alpha * x1 + (1-alpha) * x2
            loss = alpha * loss1 + (1-alpha) * loss2
            if cfg.eval_parallel:
                loss = torch.mean(loss) # DP

            #record
            pred = torch.argmax(pred,1).cpu()

            for j in range(len(subjs)):
                subj = '{:06d}'.format(subjs[j])
                if subj not in subjs_target:
                    subjs_pred[subj], subjs_target[subj] = [], []
                subjs_pred[subj].append(pred[j])
                subjs_target[subj].append(target[j])

    return subjs_target, subjs_pred


def stat_singleStages(target, pred):
    accs = ['','','','','']
    f1s = f1_score(target, pred, labels=[0,1,2,3,4], average = None)
    count = [0,0,0,0,0]
    for stg in range(5):
        stg_idx = torch.where(target == stg)[0]
        count[stg] = len(stg_idx)
        if len(stg_idx) > 0:
            accs[stg] = accuracy_score(target[stg_idx], pred[stg_idx])
    return accs, f1s.tolist(), count

def getDataset(index):
    subjIdx = { 'SHHS1':(0,5792),
                    'SHHS2':(5793,8443),
                    'CCSHS':(8444,8958),
                    'SOF':(8959,9411),
                    'CFS':(9412,10141),
                    'MROS1':(10142,13046),
                    'MROS2':(13047,14072)}
    if type(index)!=int:
        index = int(index)
    for dataset in subjIdx:
        start, end = subjIdx[dataset]
        if start <= index <= end:
            return dataset
    return ''


if __name__ == "__main__":
    from tools.config_handler import yamlStruct
    cfg = yamlStruct()
    cfg.add('dataset', ["SHHS1", "SHHS2", "CCSHS", "SOF", "CFS", "MROS1", "MROS2"])
    cfg.add('eval_parallel',True)
    cfg.add('redir_root',[r"H:\data\filtered_data\subjects", r"H:\data\filtered_data\subjects"])
    cfg.add('redir_cache',False)
    cfg.add('BATCH_SIZE',150)
    cfg.add('eval_thread',6)
    cfg.add('tvt','test')
    cfg.add('freq',125)
    cfg.add('best_ckp',r".\weights_all\checkpoint")
    savePath = r'./subjs_mixed_cohort.csv'

    # from tools.read_demographics import Demographics
    # demographics = Demographics('SHHS1')

    s_t, s_p = tester(cfg, alpha = .5)
    with open(savePath, 'w') as f:
            f.write('fileId,  Accuracy,  F1, dataset, stage, Kappa\n')
    for idx in s_t:
        st = torch.Tensor(s_t[idx])
        sp = torch.Tensor(s_p[idx])

        # stages
        acc = accuracy_score(st, sp)
        kappa = cohen_kappa_score(st, sp,labels = [0,1,2,3,4])
        mf1 = f1_score(st,sp,labels = [0,1,2,3,4], average = 'macro')
        accs, f1s, count_stage = stat_singleStages(st, sp)
        
        # save this subj (for R)
        stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
        for i in range(5):
            line = '{:},{:},{:},{:},{:},\n'.format(idx, accs[i], f1s[i], getDataset(idx), stages[i])
            with open(savePath, 'a') as f:
                f.write(line)
        line =  '{:},{:},{:},{:},Overall,{:}\n'.format(idx,  acc,  mf1, getDataset(idx), kappa)
        with open(savePath, 'a') as f:
                f.write(line)

        # # demographics
        # with open(os.path.join(cfg.redir_root[0], idx, idx + '.info'), 'r') as f:
        #     recordID = f.readlines()[0].split(',')[0]
        # info = demographics.read_demographics(recordID)
        # line += str(info.sex) + ',' + str(info.age) + ',' + str(info.ahi) + ',' + recordID + '\n'
        
        
    print('done!')