def acc_stage(target, pred):
    assert len(target) == len(pred)
    count_all = len(target)
    count_t = 0
    for i in range(len(target)):
        if target[i] not in (0,1,2,3,4):
            count_all -= 1
            continue
        count_t += 1 if target[i] == pred[i] else 0
    return float(count_t / count_all)
            
