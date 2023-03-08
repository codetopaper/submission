from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
import argparse
from datasets_clothing1m import clothing_dataset, clothing_LID
from tools import accuracy, make_determine, mle_batch
import pandas as pd
import csv 
import torch
import torch.optim as optim
import torch.nn as nn
from transformer_clothing1m import clothing_tran

import time

import torchvision.models as models


folders = ['data', 'log', 'record']
for folder in folders:
    path = os.path.join('./', folder)
    if not os.path.exists(path):
        os.makedirs(path)

#https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
class ResNet_2fc(nn.Module):
    def __init__(self, original_model):
        super(ResNet_2fc, self).__init__()
        self.features = original_model
        self.fc_layer_2 = nn.Linear(512, 14)
        
    def forward(self, x):
        lid_feature = self.features(x)
        x = self.fc_layer_2(lid_feature)
        return(x, lid_feature)

def train(class_no = 0, batch_size=128, epochs=50, random_seed=1, noise_ratio=0, dataset='clothing1m'):


    print(noise_ratio)
    if dataset=='clothing1m':
        make_determine()
        train_data = clothing_dataset(noise_ratio, random_seed, transform=clothing_tran('train'), mode='train')
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        data_length = len(train_data)

        make_determine()
        test_data = clothing_dataset(noise_ratio, random_seed, transform=clothing_tran('test'), mode='test')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        make_determine()
        resnet18 = models.resnet18(pretrained=False)
        resnet18.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        model = ResNet_2fc(resnet18)
        model.fc_layer_2.train(False) 


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    
    
    if (torch.cuda.device_count() > 1):
        model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.05)
    start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    
    print('training data length', data_length)
    N = 1280
    no_LID_sequences = 50
    LID_file = dataset +'_size_' + str(data_length) + '_indices.csv'
    if not os.path.isfile(LID_file):
        with open(LID_file, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for i in range(no_LID_sequences):
                random.seed(i)
                idx = random.sample(range(data_length), N)
                writer.writerow(np.array(idx))
        csvFile.close()
    N_indices = pd.read_csv(LID_file, header=None, index_col=None)
    #to-do 2
    if dataset=='clothing1m':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    data_points = {}
    for row in range(len(N_indices.index)): 
        data_points[row] = np.array(N_indices.iloc[row]).astype(int)

    #if not baseline:
    if noise_ratio[class_no] < 100: #not baseline
        record_file = './record/'+str(dataset)+'_'+str(noise_ratio[class_no])+'_seed_'+str(random_seed)+'_record.csv' #for initialization
    else:
        record_file = './record/'+str(dataset)+'_bl_seed_' + str(random_seed)+'_record.csv'
    header = ['epoch', 'train loss', 'train acc', 'train time', 'test loss', 'test acc', 'test time', 'LID time']
    with open(record_file, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(header)
    csvFile.close()

    for epoch in range(start_epoch, epochs):
        print('--------epoch: {}/{}--------'.format(epoch, epochs))
        #training
        record = [int(epoch), ]
        train_acc = 0.0
        train_loss = 0.0
        train_data_len = 0
        model.train()
        tr_start_time = time.time()

        for i, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            train_data_len += y_train.size(0)
            predictions, _ = model(X_train) #as weights are not updated here, the features actually belong to the previous epoch
            loss = criterion(predictions, y_train)
            acc = accuracy(predictions, y_train)
            train_acc += acc
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        record.extend([train_loss/float(train_data_len), train_acc/float(train_data_len), time.time() - tr_start_time])
        
        #validation
        if epoch%20==0 or epoch==(epochs-1):
            test_acc = 0.0
            test_loss = 0.0
            test_data_len = 0
            test_start_time = time.time()
            with torch.no_grad():
                model.eval()
                for i, (X_test, y_test) in enumerate(test_loader):
                    X_test = X_test.to(device)
                    y_test = y_test.to(device)
                    test_data_len += y_test.size(0)
                    predictions, _ = model(X_test)
                    loss = criterion(predictions, y_test)
                    acc = accuracy(predictions, y_test)
                    test_acc += acc
                    test_loss += loss.item()
            record.extend([test_loss/float(test_data_len), test_acc/float(test_data_len), time.time() - test_start_time])
        


        scheduler.step()
        
        #at the end of each epoch, compute lid scores
        lid_sequences = []
        LID_path = os.path.join('./log', str(random_seed))
        if not os.path.exists(LID_path):
            os.makedirs(LID_path)
        final_file_name = LID_path + '/lid_%s_%s_%s.csv' % \
                        (dataset, random_seed, noise_ratio[class_no])


        LID_time = time.time()
        for key in data_points:

            if dataset=='clothing1m':
                lid_data = clothing_LID(data_points[key], clothing_tran('test'))
                lid_loader = torch.utils.data.DataLoader(lid_data, batch_size=len(data_points[key]), shuffle=True, num_workers=0, drop_last=False)
            model.train()
            ft_list = []
            counter = 0
            for i, X_train in enumerate(lid_loader):
                X_train = X_train.to(device)
                with torch.no_grad():
                    _, X_act = model(X_train)
                    X_act = np.asarray(X_act.cpu().detach(), dtype=np.float32).reshape((X_act.shape[0], -1))
                    
                    if counter == 0:
                        ft_list=X_act
                        counter += 1
                    else:
                        ft_list = np.append(ft_list, X_act, axis=0)
                    
            X_act = np.asarray(ft_list)
            lids = []
            bs = 128
            s = int(X_train.shape[0]/bs)
            for ss in range(s):
                lid_batch = np.zeros(shape=(bs, 1))
                lid_batch[:, 0] = mle_batch(X_act[ss*bs:(ss+1)*bs], X_act[ss*bs:(ss+1)*bs]) 
                lids.extend(lid_batch)
            lids = np.asarray(lids, dtype=np.float32)
            lid_sequences.append(np.mean(lids))

        with open(final_file_name, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(np.array(lid_sequences))
        csvFile.close()
        record.extend([time.time()-LID_time])

        with open(record_file, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(np.array(record))
        csvFile.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int, default=6
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int, default=32
    )
    parser.add_argument(
        '-s', '--random_seed',
        help="The default 20 random seeds are: 0, 235, 905, 1286, 2048, 4096, 5192, 7813, 11247, 11946, 14557, 16860, 21347, 27718, 35697, 35715, 37330, 40526, 43412, 45270.",
        required=False, type=int
    )

    args = parser.parse_args()


    num_classes = 14
    abase = {'1010': {}, '0': {}, '300': {}, '830': {}, '840': {},\
             '850': {}, '854': {}, '858': {}, '860': {},\
             '862': {}, '864': {}, '876': {}, '877': {},\
             '878': {}, '879': {}, '880': {}, '881': {},\
             '882': {}, '883': {}, '884': {}, '885': {}, '886': {},\
             '909':{}, '911':{}, '913':{}, '914':{}, '915':{}, '916':{}}

    for a in abase:
        for i in range(num_classes):
            abase[a][i] = float(a) / 10.0
        print(abase[a])


    class_no = 0


    for a in abase:
        train(class_no, args.batch_size, args.epochs, args.random_seed, abase[a])
