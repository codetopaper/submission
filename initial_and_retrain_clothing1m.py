import pandas as pd
import numpy as np 
import os
#from sklearn.tree import DecisionTreeClassifier
#from baggingPU import BaggingClassifierPU
import copy
import csv
import math
from train_PU import LID_assmb, train_val_PU
import re
import argparse

####################################################################################################
#initial training PU models then record the training results as csv file. 
####################################################################################################

parser = argparse.ArgumentParser()	
parser.add_argument('--LID_dir', type=str, default = './log/', help="The directory saves LID sequences.")	
parser.add_argument('--training_start_epoch', type=int, default = 0, help="The starting epoch of the LID sequences to train PU models.")	
parser.add_argument('--training_end_epoch', type=int, default=6, help="The ending epoch of the LID sequences to train PU models.")	
args = parser.parse_args()

location = args.LID_dir	
start_ep = args.training_start_epoch	
end_ep = args.training_end_epoch

#random_seeds = ['0','235','905','2048','4096','5192','7813','11946','16860','35715']
random_seeds = [0, 235, 905, 1286, 2048, 4096, 5192, 7813, 11247, 11946,\
               14557, 16860, 21347, 27718, 35697, 35715, 37330, 40526, 43412, 45270]
#new_ = ['21347', '11247', '14557','27718','45270','43412','35697','40526']
record_file = './k14_initial_PU_' + str(len(random_seeds)) + '_seeds.csv'

#correspondence = {1: 86.4, 0.98: 88.0, 0.96: 88.2, 0.94: 88.4, 0.92: 88.5, 0.9: 88.6, 'init': 88.3}
correspondence = {1: 89.8, 0.98: 91.1, 0.96: 91.3, 0.94: 91.5, 0.92: 91.5, 0.9: 91.6, 'init': 91.6}
#initial_training = True
#if initial_training:
#    max_a = 88.3

#start_ep = 0
#end_ep = 6


count = 0
no_count = 0
for val in random_seeds:
    cp = copy.copy(random_seeds)
    cp.remove(val)
    while cp:
        tr1=cp[0]
        cp.remove(cp[0])
        for tr2 in cp:
            ini = [tr1, tr2, val]
            #print('tr1, tr2, val:', [tr1, tr2, val])
            records, header = train_val_PU(correspondence['init'], val, tr1, tr2, location, start_ep, end_ep)
            if records[1] >= 0.9:
                count += 1
                with open(record_file, 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    if count == 1:
                        writer.writerow(header)
                    writer.writerow(records)
                csvFile.close()
                print([tr1, tr2, val], count)
            else:
                no_count += 1
                print([tr1, tr2, val], 'rc below 0.9', no_count)



###################################################################################################
#read the initial training summary csv
#get the columns with label <=85, compute overall column sum s. 
#if s>=5 for 10 seeds, potentially high noise rate in the given dataset, use recall 0.9 or 0.92, 
#else, low noise rate, high recall, 1 or 0.98.
#return the qualified triples
#retrain them.
####################################################################################################

#read the initial training csv file
#record_file = './5_initial_PU_train_' + str(len(random_seeds)) + '_seeds.csv'
ini = pd.read_csv(record_file, header=0, index_col=False, engine='python')

#count the non-zero votes below cut-off line
cut_off = 85.0

count = 0
for c in ini.columns:
    try:
        if float(c) <= cut_off:
            #count += ini[c].sum()
            count += (ini[c] !=0).sum()
    except ValueError:
        pass
print('no. of non-zero votes before cut-off', count)

#re-define recall.
thresh = math.floor(len(random_seeds)*5/10.0) #every 10 seeds if the non-0 vote below 85 is < 5, then the given dataset should have low noise rate itself, use high recall.
if count < thresh:
    recall_list = [0.98, 0.99, 1]#high recall
else:
    recall_list = [0.9, 0.91, 0.92]#low recall
print('recall_list', recall_list)

triples = list(ini.loc[ini['recall'].isin(recall_list)].iloc[:,0]) #get the random seeds with above redefined recall
#print(triples)
recalls = list(ini.loc[ini['recall'].isin(recall_list)].iloc[:,1]) #their recalls
#print(recalls)

retrain_file = './14k_retrain_PU_' + str(len(random_seeds)) + '_seeds.csv'

count = 0
repeat = 5

for t in range(len(triples)):
    clicker = 0
    count += 1
    print(count, 'tr1, tr2, val:', triples[t], recalls[t])
    for rp in range(repeat):
        tr1, tr2, val = map(int, re.findall('\d+', triples[t]))
        #print(recalls[t])
        record, header = train_val_PU(correspondence[recalls[t]], val, tr1, tr2, location, start_ep, end_ep)
        print('new recall',record[1])
        if record[1] == recalls[t]:
            clicker += 1
            print('clicker', clicker)
    if clicker < math.ceil(repeat/2.0):
        continue
    else:
        #only record those with repeatable and qualified recalls
        if sum(record[3:])>0.: #don't count those all 0s
            with open(retrain_file, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                #if count == 1:
                writer.writerow(header)
                writer.writerow(record)
            csvFile.close()
