import warnings
import torch,os
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from torch.nn import DataParallel as DP
from tools.metrics import acc_stage


import warnings
warnings.filterwarnings("ignore")
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def tester(cfg, alpha=.5):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    saveObj = torch.load(cfg.best_ckp)
    model = saveObj['net'].cuda()
    if type(model) == DP:
        model = model.module
    if cfg.eval_parallel:
        model = DP(model)

    try:
        setName = cfg.test_dataset
        assert len(setName)!=0
    except:
        setName = cfg.dataset
    from tools.data_loader_cohort import XY_dataset_N2One as myDataset
    testSet = myDataset(tvt = cfg.tvt, frame_len = 30*cfg.freq, datasetName=setName, 
                        redir_root = cfg.redir_root, redir_cache = cfg.redir_cache)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = cfg.BATCH_SIZE*2, 
                                    shuffle = False, num_workers = cfg.eval_thread, drop_last = False)
    model.eval()

    ###
    epoch_loss = 0
    epoch_acc = 0
    record_target = torch.LongTensor([]) 
    record_pred = torch.LongTensor([])
    ###

    with torch.no_grad():
        tq = tqdm(testLoader, desc= 'Test', ncols=80, ascii=True)
        for i, (data, target) in enumerate(tq):
            inputs = Variable(data.cuda())
            targets = Variable(target.cuda())
            x1, x2, loss1, loss2 = model(inputs, targets.view([-1]).long())
            pred = alpha * x1 + (1-alpha) * x2
            loss = alpha * loss1 + (1-alpha) * loss2
            if cfg.eval_parallel:
                loss = torch.mean(loss) # DP

            #record
            pred = torch.argmax(pred,1).cpu()
            record_pred  = torch.cat([record_pred , pred])
            record_target  = torch.cat([record_target , target])

            epoch_loss += loss.item()
            epoch_acc  += accuracy_score(target, pred) 
            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss  / (i+1)), 
                        'Acc:':'{:.4f}'.format(epoch_acc  / (i+1))})
        epoch_loss  /= (i+1)

    epoch_acc  = acc_stage(record_target , record_pred ) 
    msg_loss = 'Tst Loss:{:.4f}, acc:{:.2f}\n'.format(
                epoch_loss ,  epoch_acc   * 100)
    msg_test_detail = classification_report(record_target , record_pred , labels=[0,1,2,3,4]) \
                                + str(confusion_matrix(record_target , record_pred , labels=[0,1,2,3,4])) + 'Kappa:'\
                                + str(cohen_kappa_score(record_target , record_pred , labels=[0,1,2,3,4])) + '\nF1:'\
                                + str(f1_score(record_target , record_pred , labels=[0,1,2,3,4], average='macro')) + '\n'


    print(msg_loss[:-1] + msg_test_detail)
    return msg_loss, msg_test_detail




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']= '3'
    pass