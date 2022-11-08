import os
import numpy as np
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
from data_config import channelCfg
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "UNKNOWN" :5
}
stage_dict_2 = {'5':4, '0':0, '1':1, '2':2, '3':3, '4':3, '9':5, '6':5}

def xml(path = "D:/SLEEP/shhs/polysomnography/annotations-events-profusion/shhs1/shhs1-200101-profusion.xml"):
    import xml.etree.ElementTree as ET
    anno = []
    tree = ET.ElementTree(file= path)
    root = tree.getroot()
    sleepStages = root.find('SleepStages')
    stages = sleepStages.findall('SleepStage')
    for stage in stages:
        s = stage.text
        anno.append(stage_dict_2[s])
    return anno 

# default: C3,C4,E1,E2,EMG
def getExistChn(chnNameTable, rawChns):
    chnNames = [None]*5
    for i in range(len(chnNameTable)):# C3,C4,F3,....
        if chnNameTable[i]:
            found = False
            for name in chnNameTable[i]:
                for rcn in rawChns:
                    if rcn.upper() == name.upper():
                        chnNames[i] = rcn;found = True;break
                if found:break
    return chnNames
def delExistRef(ch_names, ref_names):
    for i in range(len(ch_names)):
        if not ref_names[i]:
            continue
        if ref_names[i] in ch_names[i]:
            ref_names[i] = None

EPOCH_SEC_SIZE = 30
def read_data(edfName, annName, dataset = 'SHHS1', sampling_rate = 125, format = 'xml'):
    # load head
    unit = 1

    raw = read_raw_edf(edfName, preload=False, stim_channel=None)
    print('head ready')
    
    # get raw Sample Freq and Channel Name
    sfreq = raw.info['sfreq']
    resample = False if sfreq == sampling_rate else True
    print('【signal sampling freq】:',sfreq)
    ch_names = getExistChn(channelCfg[dataset][0], raw.ch_names)
    ref_names = getExistChn(channelCfg[dataset][1], raw.ch_names)
    if None in ch_names:
        assert 0
    delExistRef(ch_names, ref_names)
    # ch_names, ref_names ready!

    # load raw
    exclude_channel = raw.ch_names
    for cn in set(ch_names + ref_names):
        if cn is not None: 
            exclude_channel.remove(cn)
    raw = read_raw_edf(edfName, eog=(ch_names[2], ch_names[3]),
                    preload=True, stim_channel=None, exclude=exclude_channel)
    # raw.copy().plot(duration = 30, proj = False, block = True)    
    
    # preprocessing
    raw_ch_list = []
    for j in range(len(ch_names)):
        # pick channel and set reference
        if ref_names[j] == None:
            raw_chn = raw.copy().pick(ch_names[j])#copy!!!!
        else:
            raw_chn = raw.copy().pick([ch_names[j], ref_names[j]])
            raw_chn.set_eeg_reference(ref_channels=[ref_names[j]])
            raw_chn = raw_chn.pick([ch_names[j]])
        ################ Filter  ################     
        if j != 4:
            raw_chn.notch_filter([50,60], picks = (ch_names[j]))
            raw_chn.filter(l_freq = 0.3, h_freq = 35, picks = (ch_names[j]), method='iir')
        else:
            raw_chn.notch_filter([50,60], picks = (ch_names[j]))
            raw_chn.filter(l_freq = 10, h_freq = None, picks = (ch_names[j]), method='iir')
        # resample 
        if resample:
            raw_chn.resample(sampling_rate)
        data, _ = raw_chn[:]    
        ################ Unit: Volt to μV  ################      
        if np.std(data) < 1e-3:
            unit = 1e6
        if raw._orig_units[ch_names[j]] in ('µV', 'mV'):
            unit = 1e6
        data *= unit                        
        ################### Check Value ###################         
        assert 1<np.diff(np.percentile(data,[25,75]))<1e3 or 5e-2<np.std(data)<5e2 or np.max(data) - np.min(data) < 1
        ################### Done ###################
        data = np.clip(data, -500, 500)     # Clip signal to (-500, +500) 
        data = np.swapaxes(data, 0, 1)
        raw_ch_list.append(data)
        # raw_chn.copy().plot(duration = 30, proj = False, block = True)    
    for j in range(5 - 1):
        assert raw_ch_list[j].size == raw_ch_list[j+1].size
    raw_ch = np.concatenate(raw_ch_list, axis = 1) 
    assert raw_ch.shape[1]==5 and len(raw_ch.shape) == 2


    # no offset
    # assert len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) == 0
    n_epochs = len(raw_ch) // (EPOCH_SEC_SIZE * sampling_rate)
    raw_ch_slim = raw_ch[:int(n_epochs * EPOCH_SEC_SIZE * sampling_rate)]
    x = np.asarray(np.split(raw_ch_slim, n_epochs)).astype(np.float32)
    
    # No anno
    if annName == None:
        return x, None, None

    # Get anno
    if format == 'xml':
        ann = xml(annName)
    y = np.asarray(ann).astype(np.int32)
    if len(x) != len(y):
        assert 0
    
    # ensure label reasonable 
    n_stage = 0
    for i in range(5):
        n_stage += 1 if np.where(y==i)[0].shape[0] > 0 else 0
    # (anno have greater than 2 stages) AND (unkonwn < 50% epochs)
    if n_stage<3 or np.count_nonzero(y-5) < len(y) / 2:
        print('######  EXCLUDE: %s ######'%(annName))
        return x, y, 'exclude'
    
    # delete ? before and after sleep
    known_idx = np.where(y != stage_dict["UNKNOWN"])[0]
    assert len(known_idx)>=5
    start_idx = known_idx[0] 
    end_idx = known_idx[-1]
    if start_idx < 0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx+1)
    print("Data before selection: {}, {}".format(x.shape, y.shape))
    x = x[select_idx]
    y = y[select_idx]
    print("Data shape: {}, {}".format(x.shape, y.shape))

    return x, y, None
