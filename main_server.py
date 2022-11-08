from trainval_twoLoss import trainer
from tools.config_handler import YamlHandler

if __name__ == '__main__':
    yh = YamlHandler('./config/ran_shhs1_server.yaml')
    cfg = yh.read_yaml()
    try:
        netName = cfg.net
    except:
        netName = 'ResAttNet'
    
    if netName in ('ResAttNet_OneLoss', ''):
        from sleep_models.ResAtt_OneLoss import Stage_Net_E2E as myNet
    elif netName in ('ResAttNet'):
        from sleep_models.ResAtt_TwoLoss import Stage_Net_E2E as myNet
    
    trainer(myNet, cfg)
