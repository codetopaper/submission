import pandas as pd
import numpy as np 
import os
from sklearn.tree import DecisionTreeClassifier
from baggingPU import BaggingClassifierPU




def LID_assmb(seed_no, max_a, start_ep=0, end_ep=45, loc='./log'):
    '''
    returns an assembly of the LID sequences with alpha labels and loss labels, shuffle then split the LIDs and labels
    loc: the location of the LIDs
    random_seed: the csv for certain seed to be read
    max_a: the maximum alpha for each random seed's LID files
    LID data is saved in the format of: lid_dataset_(random_seed)_alpha.csv, for example, lid_cifar10_4096_86.0.csv
    or lid_cifar10_4096_101.csv for baselines (baseline has no alpha, but for notational ease, its alpha is treated as 101).
    '''
    appended = []
    loc = os.path.join(loc, str(seed_no))
    files = os.listdir(loc)
    for f in files:
        random_s = f.split('.csv')[0].split('_')[2]
        a = f.split('.csv')[0].split('_')[-1]
        if str(seed_no) == random_s:
            if float(a)<=max_a or float(a) > 100:
                T = pd.read_csv(os.path.join(loc, f), header=None, index_col=None)#T.shape[0]= epoch number, T.shape[1]=number of LID sequences
                if T.shape[0] < end_ep:
                    raise ValueError('LID length %d is shorter than the required length %d!' % (T.shape[0],end_ep))
                else:
                    T = T.transpose().iloc[:, start_ep:end_ep]#take LIDs from certain epochs only
                    no_rows = len(T.index)
                    labels = [a,]*no_rows
                    if float(a) > 100: 
                        loss_labels = np.ones(no_rows)
                    else:
                        loss_labels = np.zeros(no_rows)
                    T.insert(len(T.columns), 'labels', labels, True)
                    T.insert(len(T.columns), 'loss_labels', loss_labels, True)
                    appended.append(T)
            else:
                continue
    appended = pd.concat(appended)
    return(appended)




def train_val_PU(max_a, val, tr1, tr2, loc='./log', start_ep=0, end_ep=45):
    '''
    this function trains a triple (a combination of 3 random seeds).
    max_a: the maximum alpha used to train the PU model.
    val: the random seed number that produces the LID sequence for test
    tr1: the 1st random seed number that produces the LID sequence for training
    tr2: the 2nd random seed number that produces the LID sequence for training
    loc: the LID sequences' location
    start_ep: the starting epoch of the LID sequence
    end_ep: the ending epoch of the LID sequence
    '''
    records = [[tr1, tr2, val],]
    train1 = LID_assmb(tr1, max_a, start_ep, end_ep, loc)
    train2 = LID_assmb(tr2, max_a, start_ep, end_ep, loc)
    total = train2.append(train1)
    total = total.sample(frac=1).reset_index(drop=True) #shuffle everything
    labels = (total.iloc[:, -2]).to_numpy()
    loss_labels = (total.iloc[:, -1]).to_numpy()
    total = total.iloc[:, start_ep:end_ep]
    bc = BaggingClassifierPU(DecisionTreeClassifier(),n_estimators = 1000,max_samples = int(sum(loss_labels)),n_jobs = -1)
    bc.fit(total, loss_labels)

    v_total = LID_assmb(val, max_a, start_ep, end_ep, loc)
    v_total = v_total.sample(frac=1).reset_index(drop=True)
    v_labels = (v_total.iloc[:, -2]).to_numpy()
    v_loss_labels = (v_total.iloc[:, -1]).to_numpy()
    v_total = v_total.iloc[:, start_ep:end_ep]
    pred = bc.predict_proba(v_total)
    
    v_summary = {} 
    for i in set(v_labels):
        v_summary[i] = 0.0 
        if float(i)>100:
            bl_label = i
    for i in range(len(pred)):
        if np.isnan(pred[i][1]):
            raise ValueError('prediction has illegal value nan, please check the model and data!')
            break
        elif pred[i][1]<0.5:
            continue
        else:
            v_summary[v_labels[i]] += 1

    records.extend([v_summary[bl_label]/float(sum(v_loss_labels)), v_summary[bl_label]])
    temp = set(v_labels)
    temp.remove(bl_label)
    header = list(sorted(temp))
    header = ['train 1, 2 val', 'recall'] + [bl_label,] + header
    for r in range(3, len(header)):
        records.append(v_summary[header[r]])
    return(records, header)
    '''
    with open(record_file, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        if index == 1:
            writer.writerow(header)
        writer.writerow(records)
    csvFile.close()
    '''





