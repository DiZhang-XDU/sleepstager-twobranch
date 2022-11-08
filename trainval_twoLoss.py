import os,time
import torch
from torch.autograd import Variable
from os.path import join
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn import DataParallel as DP
from tools.metrics import acc_stage


import warnings
warnings.filterwarnings("ignore")##################################
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def trainer(myNet, cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    for p in [cfg.path['weights'], cfg.path['tblogs']]:
        if os.path.exists(p) is False:
            os.mkdir(p) 

    # tensorboard
    import shutil
    if not cfg.resume:
        shutil.rmtree(cfg.path['tblogs'])
    writer = SummaryWriter(cfg.path['tblogs'])
    
    # prepare dataloader 
    from tools.data_loader_cohort import XY_dataset_N2One as myDataset
    trainSet = myDataset(tvt = 'train', frame_len = 30*cfg.freq, datasetName=cfg.dataset, 
                        redir_root = cfg.redir_root, redir_cache = cfg.redir_cache)
    validSet = myDataset(tvt = 'valid', frame_len = 30*cfg.freq, datasetName=cfg.dataset, 
                        redir_root = cfg.redir_root, redir_cache = cfg.redir_cache)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = cfg.BATCH_SIZE, 
                                    shuffle = True, num_workers = cfg.train_thread, drop_last = False, pin_memory = True)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size = cfg.BATCH_SIZE*2, 
                                    shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = True)

    # initialization
    model = myNet(5, frame_len=30*cfg.freq).cuda()
    if cfg.resume:
        optim = torch.optim.SGD(model.parameters(), lr= 1e-1, momentum = .9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,4,1e-3)
        loadObj = torch.load(join(cfg.path['weights'], 'checkpoint_'), map_location='cpu')
        model_, epoch_, optim_, scheduler_, best_loss_val_ = loadObj['net'], loadObj['epoch'], loadObj['optim'], loadObj['sched'], loadObj['best_loss_val']
        model.load_state_dict(model_.state_dict())
        optim.load_state_dict(optim_.state_dict())
        best_loss_val, epoch = 9999, 1
    else:
        optim = torch.optim.Adam(model.parameters(), lr= cfg.LR)
        scheduler = eval(cfg.scheduler)
        best_loss_val, epoch = 9999, 1
    if cfg.train_parallel:
        model = DP(model)
        
    print('start epoch')
    step = 0
    trainIter = iter(trainLoader)   
    for epoch in range(epoch, cfg.EPOCH_MAX + 1): 
        tic = time.time()
        alpha = 1 - (epoch / cfg.EPOCH_MAX) ** 2
        name = ('train', 'valid')
        epoch_loss = {i:0 for i in name}
        epoch_acc = {i:0 for i in name}
        epoch_mf1 = {i:0 for i in name}
        eval_EachLoss = {1:[], 2:[]}
        eval_EachPred = {1:[], 2:[]}
        record_target = {i:torch.LongTensor([]) for i in name}
        record_pred = {i:torch.LongTensor([]) for i in name}

        torch.cuda.empty_cache()
        model.train()
        
        tq = tqdm(range(cfg.EPOCH_STEP), desc= 'Trn', ncols=80, ascii=True)
        for i, _ in enumerate(tq):
            data, target = next(trainIter)
            step += 1
            if step == len(trainLoader):
                step = 0
                trainIter = iter(trainLoader)

            inputs = Variable(data.cuda())
            targets = Variable(target.cuda())

            inputs.requires_grad = True
            # forward
            x1, x2, loss1, loss2 = model(inputs, targets.view([-1]).long())
            pred = alpha * x1 + (1 - alpha) * x2
            loss = alpha * loss1 + (1 - alpha) * loss2
            if cfg.train_parallel:
                loss = torch.mean(loss)
            
            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # record
            pred = torch.argmax(pred,1).cpu()
            record_pred['train'] = torch.cat([record_pred['train'], pred])
            record_target['train'] = torch.cat([record_target['train'], target])
            
            epoch_loss['train'] += loss.item()
            epoch_acc['train'] += accuracy_score(target, pred) 
            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['train'] / (tq.n+1)), 
                            'Acc:':'{:.4f}'.format(epoch_acc['train'] / (tq.n+1))})
        epoch_loss['train'] /= (i+1)

        # eval
        torch.cuda.empty_cache()
        model_eval = DP(model) if cfg.eval_parallel and not cfg.train_parallel else model
        model_eval.eval()
        
        tq = tqdm(validLoader, desc = 'Val', ncols=75, ascii=True)
        for i, (data, target) in enumerate(tq):
            inputs = Variable(data.cuda())
            targets = Variable(target.cuda())
            with torch.no_grad(): 
                x1, x2, loss1, loss2 = model_eval(inputs, targets.view([-1]).long())
            alpha = 0.5
            pred = alpha * x1 + (1 - alpha) * x2
            loss = alpha * loss1 + (1 - alpha) * loss2
            if cfg.eval_parallel:
                loss = torch.mean(loss)
            
            # record
            pred = torch.argmax(pred,1).cpu()
            record_pred['valid'] = torch.cat([record_pred['valid'], pred])
            record_target['valid'] = torch.cat([record_target['valid'], target])

            epoch_loss['valid'] += loss.item()
            epoch_acc['valid'] += accuracy_score(target, pred) 
            
            # separately record
            eval_EachLoss[1] += loss1.cpu() if cfg.eval_parallel else [loss1.cpu()]
            eval_EachLoss[2] += loss2.cpu() if cfg.eval_parallel else [loss2.cpu()]
            eval_EachPred[1] += torch.argmax(x1,1).cpu()
            eval_EachPred[2] += torch.argmax(x2,1).cpu()
            

            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['valid'] / (i+1)), 
                        'Acc:':'{:.4f}'.format(epoch_acc['valid'] / (i+1))})
                        
        epoch_loss['valid'] /= (i+1)
        
        # epoch end
        scheduler.step()
        for idx in name:
            epoch_acc[idx] = acc_stage(record_target[idx], record_pred[idx])
            epoch_mf1[idx] = f1_score(record_target[idx], record_pred[idx], labels=[0,1,2,3,4], average='macro')

        msg_epoch = 'epoch:{:02d}, time:{:2f}\n'.format(epoch, time.time() - tic)
        msg_loss = 'Trn Loss:{:.4f}, acc:{:.2f}  Val Loss:{:.4f}, acc:{:.2f}\n'.format(
            epoch_loss['train'], epoch_acc['train'] * 100, 
            epoch_loss['valid'], epoch_acc['valid'] * 100)
        
        msg_detail = classification_report(record_target['valid'], record_pred['valid'], labels=[0,1,2,3,4]) \
                                 + str(confusion_matrix(record_target['valid'], record_pred['valid'], labels=[0,1,2,3,4])) + '\nKappa:'\
                                 + str(cohen_kappa_score(record_target['valid'], record_pred['valid'], labels=[0,1,2,3,4])) + '\nF1:'\
                                 + str(epoch_mf1['valid']) + '\n\n'
        print(msg_epoch + msg_loss[:-1] + msg_detail)

        # save
        writer.add_scalars('Loss',{'train':epoch_loss['train'] , 'valid':epoch_loss['valid']},epoch)
        writer.add_scalars('Acc',{'train':epoch_acc['train'], 'valid':epoch_acc['valid']},epoch)
        writer.add_scalars('MF1',{'train':epoch_mf1['train'], 'valid':epoch_mf1['valid']},epoch)
        writer.add_scalars('validLoss',{'all':epoch_loss['valid'],\
                                        'Loss1':sum(eval_EachLoss[1])/len(eval_EachLoss[1]), \
                                        'Loss2':sum(eval_EachLoss[2])/len(eval_EachLoss[2])}, epoch)
        writer.add_scalars('validMF1', {'all':epoch_mf1['valid'],\
                                        'x1':f1_score(record_target['valid'], eval_EachPred[1], labels=[0,1,2,3,4], average='macro'),\
                                        'x2':f1_score(record_target['valid'], eval_EachPred[2], labels=[0,1,2,3,4], average='macro')},epoch)
        writer.add_scalars('validAcc', {'all':epoch_acc['valid'],\
                                        'x1':acc_stage(record_target['valid'], eval_EachPred[1]),\
                                        'x2':acc_stage(record_target['valid'], eval_EachPred[2])},epoch)
        with open(cfg.path['log'], 'a') as f:
            f.write(msg_epoch)
            f.write(msg_loss)
            f.write(msg_detail)
        if best_loss_val > epoch_loss['valid']:
            best_loss_val = epoch_loss['valid']
            saveObj = {'net': model, 'epoch':epoch, 'optim':optim , 'sched':scheduler, 'best_loss_val':best_loss_val}
            torch.save(saveObj, join(cfg.path['weights'], 'checkpoint'))
        torch.save(saveObj, join(cfg.path['weights'], 'epoch_{:02d}_val_loss={:4f}_acc={:.4f}'.format(epoch, epoch_loss['valid'], epoch_acc['valid'])))
            
    writer.close()

if __name__ == "__main__":
    from sleep_models.ResAtt_TwoLoss import Stage_Net_E2E
    trainer(myNet = Stage_Net_E2E, resume = False)