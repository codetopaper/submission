#some of the code is modified from https://github.com/xiaoboxia/T-Revision

import numpy as np
import argparse
import csv
import scipy.stats
import math
import tools 
import os
import torchvision.models as models
import prior_data_load
import shutil

import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer import transform_train, transform_test,transform_target
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--noise_rate', type = float, help = 'noise rate, should be less than 1')
parser.add_argument('--form', type=str)#pw or sym
parser.add_argument('--prior_path', type=str)
parser.add_argument('--retrain_file_path', type=str)
parser.add_argument('--clean_label_path', type=str, default='cifar10_data/train_labels.npy')
parser.add_argument('--image_path', type=str, default='cifar10_data/train_images.npy')
parser.add_argument('--test_label_path', type=str, default='cifar10_data/test_labels.npy')
parser.add_argument('--test_image_path', type=str, default='cifar10_data/test_images.npy')
parser.add_argument('--save_path', type=str, default='given_labels', help='the directory to save noisy labels.')
parser.add_argument('--matrix_dir', type=str, default='matrix_estimations', help='the drecotory to save estimated matrices.')
parser.add_argument('--prior_dir', type=str, default='avg_prior', help='the drecotory to save average prior, if it is not precomputed.')

parser.add_argument('--seed', type=int, default=1, help='the random seed to shuffle data')


parser.add_argument('--temp_dir', type=str, default='./temp', help='the drecotory to save temparary data.')
parser.add_argument('--prior_epochs', type = int, default=20)
parser.add_argument('--number_of_priors', type = int, default=5)

args = parser.parse_args()

num_classes = 10





#the gournd truth noise transition matrix
True_P = tools.true_p(args.form, args.noise_rate)

name = args.form+str(int(float(args.noise_rate)*100.))
if args.prior_path == None: #will have to compute prior matrices

    #get noisy labels
    noisy_label_path = args.save_path + '/' + 'cifar10_intact_' + name + '.npy'
    if os.path.isfile(noisy_label_path):
        pass
    else:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        noisy_labels = tools.noisify(args.clean_label_path, args.form, args.noise_rate)
        np.save(noisy_label_path, noisy_labels)

    #to train the prior
    lr = 0.01
    weight_decay = 1e-4
    batch_size = 128
    percentile = 97
    estimate_state = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    priors = []
    for r in range(args.number_of_priors):
        train_data = prior_data_load.cifar10_dataset(args.temp_dir, name, args.image_path, noisy_label_path, True, \
            transform=transform_train('cifar10'), target_transform=transform_target,\
                                         random_seed=(args.seed+r))
        val_data = prior_data_load.cifar10_dataset(args.temp_dir, name, args.image_path, noisy_label_path, False, \
            transform=transform_test('cifar10'), target_transform=transform_target,
                                       random_seed=(args.seed+r))
        #data_loader
        train_loader = DataLoader(dataset=train_data, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          drop_last=False)

        estimate_loader = DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2,
                             drop_last=False)

        val_loader = DataLoader(dataset=val_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2,
                        drop_last=False)

        #loss
        loss_func_ce = nn.CrossEntropyLoss()

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)

        if (torch.cuda.device_count() > 1):
            print(torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

        #estimate transition matrix
        index_num = int(len(train_data) / batch_size)
        A = np.zeros((args.prior_epochs, len(train_data), num_classes))   
        val_acc_list = []
        total_index =  index_num + 1

        for epoch in range(args.prior_epochs):
      
            print('epoch {}'.format(epoch + 1))
            model.train()
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
        
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = loss_func_ce(out, batch_y)
                train_loss += loss.item()
                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()
                loss.backward()
                optimizer.step()
        
            #torch.save(model.state_dict(), model_save_dir + '/'+ 'epoch_%d.pth'%(epoch+1,))
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data))*batch_size, train_acc / (len(train_data))))
        
            with torch.no_grad():
                model.eval()
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    out = model(batch_x)
                    loss = loss_func_ce(out, batch_y)
                    val_loss += loss.item()
                    pred = torch.max(out, 1)[1]
                    val_correct = (pred == batch_y).sum()
                    val_acc += val_correct.item()
                
            print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*batch_size, val_acc / (len(val_data)))) 
            val_acc_list.append(val_acc / (len(val_data)))
        
            with torch.no_grad():
                model.eval()
                for index,(batch_x,batch_y) in enumerate(estimate_loader):
                     batch_x = batch_x.to(device)
                     out = model(batch_x)#, revision=False)
                     out = F.softmax(out,dim=1)
                     out = out.data.cpu().numpy()
                     if index <= index_num:
                        A[epoch][index*batch_size:(index+1)*batch_size, :] = out 
                     else:
                         A[epoch][index_num*batch_size, len(train_data), :] = out 
       
        val_acc_array = np.array(val_acc_list)
        model_index = np.argmax(val_acc_array)
        print('the epoch with the best val acc:', model_index)
    
        best = A[model_index, :, :]
        y_train = np.load(args.temp_dir+'/90pcnoisytrainlb_'+name+'.npy')
        indices_list = {}
        prior = np.empty((num_classes, num_classes))
        for c in range(num_classes):
            indices_list[c] = [x for x in range(len(y_train)) if y_train[x] == c]
            temp = best[indices_list[c], :]
            #print(np.mean(temp, axis=0, keepdims=True))
            prior[c, :] = np.mean(temp, axis=0, keepdims=True)
            #for j in range(args.num_classes):
            #    Q[c, j] = np.mean(temp[:, j])
        priors.append(prior)
        print('finished estimating prior', r)
    
    if not os.path.exists(args.prior_dir):
        os.makedirs(args.prior_dir)
    prior_name = args.prior_dir+'/'+name+'.npy'
    np.save(prior_name, priors)
    args.prior_path = prior_name
    shutil.rmtree(args.temp_dir)

if args.retrain_file_path ==  None: #will have to compute retrain csv
    assert False, 'Please compute retrain file first!'
   
if args.prior_path != None and args.retrain_file_path !=  None:

    if not os.path.exists(args.matrix_dir):
        os.makedirs(args.matrix_dir)
    #the matrix estimation
    with open(args.retrain_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        lines=[r for r in reader]
        headers = lines[0::2]
        records = lines[1::2]
    csvfile.close()

    prior_matrix=np.load(args.prior_path)

    matrix_est = []
    for pp in range(len(prior_matrix)):
        intermediate_Q = {}
        for row in range(len(headers)):
            header = headers[row]
            record = records[row]
            record[1:] = [float(r) for r in record[1:] if r != '']
            m = max(record[3:])
            if record[1] not in intermediate_Q.keys():
                intermediate_Q[float(record[1])] = [] 
            indices = [l for l in range(3,len(record)) if record[l] == m] #if there are more than 1 entries have the max vote
        
            #get one intermediate estimate from one record
            Q_tops = np.zeros((num_classes,num_classes))
            for idx in range(len(indices)):
            
                #get the top vote's alpha
                alpha_best = float(header[indices[idx]])
                #print(base_alpha_vector)
                alpha_vector = tools.alpha_v(alpha_best, num_classes)
            
                #record[1] is the recall of that retrained
                e_p, e_pp = tools.epsilons(record[1], num_classes)#, N)
            
                #get one estimate for one max vote
                Q = tools.Q_estima(e_p, e_pp, alpha_vector, prior_matrix[pp], num_classes)

                Q_tops += Q
            Q_tops /= float(len(indices))
            intermediate_Q[float(record[1])].append(Q_tops)

        #get the average according to recall
        to_avg=[]
        for ke in intermediate_Q.keys():
            to_avg.append(np.average([m for m in intermediate_Q[ke]], axis=0))
        final_Q = np.average(to_avg, axis=0)
        

        matrix_est.append(final_Q)
    np.save(args.matrix_dir+'/'+name+'.npy', matrix_est)

    #for now we assume the dataset is intact
    class_ratio = np.ones(num_classes)/float(num_classes)

    KL_losses = []
    for q in range(len(matrix_est)):
        KL_loss = tools.fwd_KL(True_P, matrix_est[q], class_ratio)
        KL_losses.append(KL_loss)

    print('the average KL loss is:', np.mean(KL_losses))





