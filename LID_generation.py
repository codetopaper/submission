#some of the code is modified from https://github.com/xiaoboxia/T-Revision
#and https://github.com/xingjunm/dimensionality-driven-learning/blob/master/train_models.py

from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
import argparse
from prior_data_load import cifar10_train, cifar10_test, cifar10_LID, manipulate_labels

from models import cifar_10_CNN

from tools import accuracy, make_determine, mle_batch, noisify
import pandas as pd
import csv 
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from transformer import transform_test, transform_target, no_rc_transform_train, transform_train

import time




folders = ['data', 'log', 'record']
for folder in folders:
    path = os.path.join('./', folder)
    if not os.path.exists(path):
        os.makedirs(path)


def train(train_img, given_dataset_labels, noise_ratio, class_no = 0, random_seed=1, dataset='cifar10', batch_size=128, epochs=45,\
 test_data='cifar10_data/test_images.npy', test_lb='cifar10_data/test_labels.npy'):


    if dataset =='cifar10':
        #synthesize alpha-increments or baseline
        noisy_lb = manipulate_labels(given_dataset_labels, dataset, noise_ratio, class_no, random_seed)

        make_determine()
        train_data = cifar10_train(train_img, noisy_lb, no_rc_transform_train(dataset), transform_target)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        data_length = len(train_data)

        make_determine()
        test_data = cifar10_test(test_data, test_lb, transform_test(dataset), transform_target)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

        make_determine()
        model = cifar_10_CNN()
        model.fc_layer_2.train(False) 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if (torch.cuda.device_count() > 1):
        model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    start_epoch = 0
    
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

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
    if dataset=='cifar10':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)
        data_points = {}
    for row in range(len(N_indices.index)): 
        data_points[row] = np.array(N_indices.iloc[row]).astype(int)


    if noise_ratio[class_no] < 100: 
        record_file = './record/'+str(dataset)+'_'+str(noise_ratio[class_no])+'_seed_'+str(random_seed)+'_record.csv' #for initialization
    else:
        record_file = './record/'+str(dataset)+'_bl_seed_' + str(random_seed)+'_record.csv'
    header = ['epoch', 'train loss', 'train acc', 'train time', 'test loss', 'test acc', 'test time']
    with open(record_file, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(header)
    csvFile.close()

    for epoch in range(start_epoch, epochs):
        #print('--------epoch: {}/{}--------'.format(epoch, epochs))
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
            predictions, _ = model(X_train) 
            loss = criterion(predictions, y_train)
            acc = accuracy(predictions, y_train)
            train_acc += acc
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        record.extend([train_loss/float(train_data_len), train_acc/float(train_data_len), time.time() - tr_start_time])        
        
        #test
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
        
        with open(record_file, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(np.array(record))
        csvFile.close()

        scheduler.step()
        
        #at the end of each epoch, compute lid scores
        lid_sequences = []
        LID_path = os.path.join('./log', str(random_seed))
        if not os.path.exists(LID_path):
            os.makedirs(LID_path)
        final_file_name = LID_path + '/lid_%s_%s_%s.csv' % \
                        (dataset, random_seed, noise_ratio[0])

        for key in data_points:
            lids = []
            if dataset=='cifar10':
                lid_data = cifar10_LID(train_img, data_points[key], transform_test(dataset))
                lid_loader = torch.utils.data.DataLoader(lid_data, batch_size=len(data_points[key]), shuffle=True, num_workers=2, drop_last=False)
            model.train()

        
            for i, X_train in enumerate(lid_loader):
                X_train = X_train.to(device)
                with torch.no_grad():
                    _, X_act = model(X_train)
                    X_act = np.asarray(X_act.cpu().detach(), dtype=np.float32).reshape((X_act.shape[0], -1))

            s = int(X_train.shape[0]/batch_size)
            for ss in range(s):
                lid_batch = np.zeros(shape=(batch_size, 1))
                lid_batch[:, 0] = mle_batch(X_act[ss*batch_size:(ss+1)*batch_size], X_act[ss*batch_size:(ss+1)*batch_size]) 
                lids.extend(lid_batch)
            lids = np.asarray(lids, dtype=np.float32)
            lid_sequences.append(np.mean(lids))

        with open(final_file_name, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(np.array(lid_sequences))
        csvFile.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images', type=str, default = 'cifar10_data/train_images.npy', help="The training images of the dataset.")
    parser.add_argument('--clean_label_path', type=str, default = 'cifar10_data/train_labels.npy', help="The training images of the dataset.")
    parser.add_argument('--save_path', type=str, default='given_labels', help='the directory to save noisy labels.')
    parser.add_argument('--noise_rate', type = float, required=True, help = 'noise rate, should be less than 1')
    parser.add_argument('--form', type=str, required=True, help = 'pw (pairwise) or sym (symmetric)')#pw or sym
    parser.add_argument('--dataset', type=str, default='cifar10', help = 'dataset to work on')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--test_images', type=str, default= 'cifar10_data/test_images.npy', help="The test images of the dataset.")
    parser.add_argument('--test_labels', type=str, default= 'cifar10_data/test_labels.npy', help="The test labels of the dataset.")


    args = parser.parse_args()

    name = args.form+str(int(float(args.noise_rate)*100.))
    given_dataset_labels = args.save_path + '/' + 'cifar10_intact_' + name + '.npy'
    if os.path.isfile(given_dataset_labels):
        pass
    else:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        noisy_labels = noisify(args.clean_label_path, args.form, args.noise_rate)
        np.save(given_dataset_labels, noisy_labels)

       
    num_classes = 10
    #uniform alpha vectors
    abase = {'1010':{}, '0':{},'300':{}, '830':{}, '840':{},\
             '850':{}, '854':{}, '858':{}, '860':{}, \
             '862':{},  '864':{}, '876':{}, '877':{}, \
             '878':{}, '879':{}, '880':{}, '881':{},\
             '882':{}, '883':{}, '884':{}, '885':{}, '886':{}}
    
    for a in abase:
        for i in range(num_classes):
            abase[a][i]=float(a)/10.0
        print(abase[a])

    class_no = 0

    #the preset 10 random seeds
    for seed in [0, 235, 905, 2048, 4096, 5192, 7813, 11946, 16860, 35715]:
        print('Gathering LID sequences for random seed', seed)
        for a in abase:
            train(args.train_images, given_dataset_labels, abase[a], class_no, seed, args.dataset, args.batch_size, args.epochs, args.test_images, args.test_labels)
