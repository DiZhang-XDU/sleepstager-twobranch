class dirCfg():
    def __init__(self, path, nid2recNID = '', nid2annKey = '', format = '.xml'):
        self.path = path
        self.nid2recNID = nid2recNID
        self.nid2annKey = nid2annKey
        self.format = format
cfgData = {
    'SHHS1':dirCfg(r'i:\sleepdata\shhs\polysomnography\edfs\shhs1', nid2recNID = 'fileIDs[i][6:12]'),
    'SHHS2':dirCfg(r'i:\sleepdata\shhs\polysomnography\edfs\shhs2', nid2recNID = 'fileIDs[i][6:12]'),
    'CCSHS':dirCfg(r'i:\sleepdata\ccshs\polysomnography\edfs', nid2recNID = 'fileIDs[i][11:18]'),
    'SOF':dirCfg(r'i:\sleepdata\sof\polysomnography\edfs', nid2recNID = 'fileIDs[i][12:17]'),
    'CFS':dirCfg(r'i:\sleepdata\cfs\polysomnography\edfs', nid2recNID = 'fileIDs[i][11:17]'),
    'MROS1':dirCfg(r'i:\sleepdata\mros\polysomnography\edfs\visit1', nid2recNID = '(fileIDs[i][12:18]).upper()'),
    'MROS2':dirCfg(r'i:\sleepdata\mros\polysomnography\edfs\visit2', nid2recNID = '(fileIDs[i][12:18]).upper()'),
}
# (path, depth, nid2recNID, format)
cfgAnno = {
    'Custum':dirCfg(None),
    'SHHS1':dirCfg(r'i:\sleepdata\shhs\polysomnography\annotations-events-profusion\shhs1',
                    nid2annKey="fileIDs[i]", format='xml' ),
    'SHHS2':dirCfg(r'i:\sleepdata\shhs\polysomnography\annotations-events-profusion\shhs2',
                    nid2annKey="fileIDs[i]", format='xml' ),
    'CCSHS':dirCfg(r'i:\sleepdata\ccshs\polysomnography\annotations-events-profusion', 
                    nid2annKey="fileIDs[i]", format='xml' ),
    'SOF':dirCfg(r'i:\sleepdata\sof\polysomnography\annotations-events-profusion', 
                    nid2annKey="fileIDs[i]", format='xml' ),
    'CFS':dirCfg(r'i:\sleepdata\cfs\polysomnography\annotations-events-profusion',
                    nid2annKey="fileIDs[i]", format='xml' ),
    'MROS1':dirCfg(r'i:\sleepdata\mros\polysomnography\annotations-events-profusion\visit1',
                    nid2annKey="fileIDs[i]", format='xml' ),
    'MROS2':dirCfg(r'i:\sleepdata\mros\polysomnography\annotations-events-profusion\visit2',
                    nid2annKey="fileIDs[i]", format='xml' )
}

# channel
channelCfg = {
    'SHHS1':((('EEG(sec)', 'EEG2','EEG 2','EEG(SEC)','EEG sec'), ('EEG',), ('EOG(L)',), ('EOG(R)',), ('EMG',)),
            [None, None, None, None, None]),
    'SHHS2':((('EEG(sec)', 'EEG2'), ('EEG',), ('EOG(L)',), ('EOG(R)',), ('EMG',)),
            [None, None, None, None, None]),
    'CCSHS':((('C3',), ('C4',), ('LOC',), ('ROC',), ('EMG1',)),
            (('A2',), ('A1',), ('A2',), ('A2',), ('EMG2',))),
    'SOF':((('C3',), ('C4',), ('LOC',), ('ROC',), ('L Chin','EMG/L')),
            (('A2',), ('A1',), ('A2',), ('A2',), ('R Chin', 'EMG/R'))),
    'CFS':((('C3',), ('C4',), ('LOC',), ('ROC',), ('EMG1',)),
            (('M2',), ('M1',), ('M2',), ('M2',), ('EMG2',))),
    'MROS1':((('C3', 'C3-A2'), ('C4', 'C4-A1'), ('LOC',), ('ROC',), ('L Chin', 'L Chin-R Chin')),
            (('A2', ), ('A1', ), ('A2', ), ('A2', ), ('R Chin',))),
    'MROS2':((('C3',), ('C4',), ('E1','LOC',), ('E2','ROC',), ('LChin','L Chin')),
            (('M2', ), ('M1', ), ('M2', ), ('M2', ), ('RChin','R Chin'))),
}
