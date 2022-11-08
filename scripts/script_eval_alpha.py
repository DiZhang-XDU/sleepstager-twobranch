import warnings
import torch,os
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import dataset
from tqdm import tqdm
import numpy as np
from tools.data_loader_cohort_stable import XY_dataset_N2One as myDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from torch.nn import DataParallel as DP
from tools.metrics import acc_stage

# os.environ['CUDA_VISIBLE_DEVICES']= '1'
os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
import warnings
warnings.filterwarnings("ignore")##################################
# !!!!!!!!!!!!!!!!! DataLoader 
import multiprocessing
multiprocessing.set_start_method('spawn', True)


# def tester(testSet = myDataset, weightDir = 'weights/checkpoint',  alpha = 0.5, dp = True):
def tester(cfg, alpha=.5):
    # from sleep_models.ResAtt_TwoLoss import Stage_Net_E2E as myNet
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
    record_stable_target = torch.LongTensor([]) 
    record_stable_pred = torch.LongTensor([])
    record_transfer_target = torch.LongTensor([]) 
    record_transfer_pred = torch.LongTensor([])
    ###

    with torch.no_grad():
        tq = tqdm(testLoader, desc= 'Test', ncols=80, ascii=True)
        for i, (data, target, stable) in enumerate(tq):
            inputs = Variable(data.cuda())
            targets = Variable(target.cuda())
            x1, x2, loss1, loss2 = model(inputs, targets.view([-1]).long())
            pred = alpha * x1 + (1-alpha) * x2
            loss = alpha * loss1 + (1-alpha) * loss2
            if cfg.eval_parallel:
                loss = torch.mean(loss) # DP

            #record
            pred = torch.argmax(pred,1).cpu()
            stable = stable.view([-1])
            transfer = ~stable
            record_stable_pred  = torch.cat([record_stable_pred , pred[stable]])
            record_stable_target  = torch.cat([record_stable_target , target[stable]])
            record_transfer_pred  = torch.cat([record_transfer_pred , pred[transfer]])
            record_transfer_target  = torch.cat([record_transfer_target , target[transfer]])



    msg_stable_detail = classification_report(record_stable_target , record_stable_pred , labels=[0,1,2,3,4]) \
                                + str(confusion_matrix(record_stable_target , record_stable_pred , labels=[0,1,2,3,4])) + 'Kappa:'\
                                + str(cohen_kappa_score(record_stable_target , record_stable_pred , labels=[0,1,2,3,4])) + '\nF1:'\
                                + str(f1_score(record_stable_target , record_stable_pred , labels=[0,1,2,3,4], average='macro')) + '\n'
    msg_transfer_detail = classification_report(record_transfer_target , record_transfer_pred , labels=[0,1,2,3,4]) \
                                + str(confusion_matrix(record_transfer_target , record_transfer_pred , labels=[0,1,2,3,4])) + 'Kappa:'\
                                + str(cohen_kappa_score(record_transfer_target , record_transfer_pred , labels=[0,1,2,3,4])) + '\nF1:'\
                                + str(f1_score(record_transfer_target , record_transfer_pred , labels=[0,1,2,3,4], average='macro')) + '\n'
    st_acc = acc_stage(record_stable_target , record_stable_pred)
    st_k = cohen_kappa_score(record_stable_target , record_stable_pred , labels=[0,1,2,3,4])
    st_mf1 = f1_score(record_stable_target , record_stable_pred , labels=[0,1,2,3,4], average='macro')
    tr_acc = acc_stage(record_transfer_target , record_transfer_pred)
    tr_k = cohen_kappa_score(record_transfer_target , record_transfer_pred , labels=[0,1,2,3,4])
    tr_mf1 = f1_score(record_transfer_target , record_transfer_pred , labels=[0,1,2,3,4], average='macro')


    print('alpha=', alpha)
    print(msg_stable_detail + msg_transfer_detail + '\n')
    return (st_acc, st_k, st_mf1, tr_acc, tr_k, tr_mf1)




if __name__ == "__main__":
    
    from tools.config_handler import yamlStruct
    cfg = yamlStruct()
    cfg.add('dataset','SHHS1')
    cfg.add('eval_parallel',True)
    cfg.add('redir_root',False)
    cfg.add('redir_cache',False)
    cfg.add('BATCH_SIZE',200)
    cfg.add('eval_thread',10)
    cfg.add('tvt','test')
    cfg.add('freq',125)
    cfg.add('best_ckp',r'.paperwork3\shhs1_2022\weights_client\epoch_31_val_loss=0.353225_acc=0.8820')

    results = [[],[],[],[],[],[]]
    for alpha in range(50, 55, 5):
        result = tester(cfg, alpha = alpha/100)
        for i in range(6):
            results[i].append(result[i])
    print(results)
    print('done!')