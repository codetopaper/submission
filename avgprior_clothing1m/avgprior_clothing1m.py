import torch
import numpy as np
import data_load_clothing1m
import argparse

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from transformer_clothing1m import clothing_tran, transform_target
import csv
import pandas as pd

import torchvision.models as models

import os

#from main_both import fwd_KL

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'clothing1m')
parser.add_argument('--n_epoch_estimate', type=int, default=10)
parser.add_argument('--num_classes', type=int, default=14)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--ground_truth', type=str, default='../../../clothing1m/images/clothing1m_matrix_samples_matrix.csv')
parser.add_argument('--data_dir', type=str, default='../../../clothing1m/images/matrix_subset_noisy_clean.txt')





class_ratio = [0.07575852,0.07797022,0.03878263,0.10538284,0.05463834,0.0772849,\
 0.05778456,0.03062115,0.08363965,0.08865491,0.07404523,0.07610118,\
 0.06990219,0.08943368] #32k





def fwd_KL(P, Q, class_ratio, fill_zero=True):
    P = np.asarray(P)  # the ground_truth matrix
    Q = np.asarray(Q)  # the estimate
    #P /= P.sum(axis=1, keepdims=True) #this can make a difference
    Q /= Q.sum(axis=1, keepdims=True)
    # assert P.shape[0] == P.shape[1]
    if P.shape[0] != P.shape[1]:
        raise ValueError('The first matrice should be square!')
    elif Q.shape[0] != Q.shape[1]:
        raise ValueError('The second matrice should be square!')
    elif P.shape[0] != Q.shape[0]:
        raise ValueError('The dimension of the 2 input matrices are different!')
    elif P.shape[0] != len(class_ratio):
        raise ValueError('The dimension of the input matrice is different from the class ratio vector!')
    else:
        if fill_zero == True:
            Q = np.where(Q == 0., 1.e-8, Q)
        l = P * np.log2(P / Q)
        l = np.where(np.isnan(l), 0., l)  # loss for entries in P with value 0 will be nan, need to replace it.
        l = l.sum(axis=1)
        class_ratio = np.asarray(class_ratio)
        loss = np.dot(class_ratio, l)

    return (loss)





#main function
def main(r):
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    True_T = pd.read_csv(args.ground_truth, header=0, index_col=0, engine='python').iloc[0:args.num_classes, 0:args.num_classes]

    name = '32k_bs'+str(args.batch_size)+'epes'+str(args.n_epoch_estimate)


    train_data = data_load_clothing1m.clothing1m_dataset(args.data_dir, name+str(r),  True, transform=clothing_tran('train'), target_transform=transform_target,random_seed=r)
    print('train_data length',len(train_data))
    val_data = data_load_clothing1m.clothing1m_dataset(args.data_dir, name+str(r), False, transform=clothing_tran('test'), target_transform=transform_target,random_seed=r)
    print('val_data length',len(val_data))

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, args.num_classes)

    
    #optimizer and StepLR
    optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler_es = MultiStepLR(optimizer_es, milestones=[5, 10], gamma=0.1)

    
    #data_loader
    train_loader = DataLoader(dataset=train_data, 
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=False)

    estimate_loader = DataLoader(dataset=train_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    val_loader = DataLoader(dataset=val_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False)



    #loss
    loss_func_ce = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    
    model = model.to(device)
    loss_func_ce = loss_func_ce.to(device)

    index_num = int(len(train_data) / args.batch_size)
    A = torch.zeros((args.n_epoch_estimate, len(train_data), args.num_classes))   
    val_acc_list = []
    #total_index = index_num + 1


    
    #print('Estimate transition matirx......Waiting......')
    #best_acc = 0.
    for epoch in range(args.n_epoch_estimate):
      
        print('epoch {}'.format(epoch + 1))
        model.train()
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
     
        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer_es.zero_grad()
            out = model(batch_x)
            loss = loss_func_ce(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_es.step()
        
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size, train_acc / (len(train_data))))
        
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
                
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data)))) 
        val_acc_list.append(val_acc / (len(val_data)))


        with torch.no_grad():
            model.eval()
            for index,(batch_x,batch_y) in enumerate(estimate_loader):
                batch_x = batch_x.to(device)
                out = model(batch_x)
                out = F.softmax(out,dim=1)
                out = out.cpu()
                if index <= index_num:
                    A[epoch][index*args.batch_size:(index+1)*args.batch_size, :] = out 
                else:
                    A[epoch][index_num*args.batch_size, len(train_data), :] = out

    scheduler_es.step()

    val_acc_array = np.array(val_acc_list)
    model_index = np.argmax(val_acc_array)
    print('the epoch with the best val acc:', model_index)
    

    best = A[model_index, :, :].numpy()
    print('best shape', best.shape)
    y_train = np.load('90pctrainlb_'+name+str(r)+'.npy')
    indices_list = {}
    Q = np.empty((args.num_classes, args.num_classes))
    for c in range(args.num_classes):
        indices_list[c] = [x for x in range(len(y_train)) if y_train[x] == c]
        temp = best[indices_list[c], :] #as shuffle = False
        Q[c, :] = np.mean(temp, axis=0, keepdims=True)
    Q /= Q.sum(axis=1, keepdims=True)

    initial_loss = fwd_KL(True_T, Q, class_ratio)
    with open('losses_' + name + '.csv', 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(np.array([initial_loss,]))
    csvFile.close()

    os.remove('90pctrainlb_'+name+str(r)+'.npy')

    return(Q, initial_loss, name)


if __name__=='__main__':

    matrices = []
    initial_losses = []
    for r in range(5):
        m, inil, name = main(r)
        initial_losses.append(inil)
        matrices.append(m)
        print('finish run', r)
        print('-------------')

    np.save('avgprior_'+name+'.npy', matrices)
    with open('losses_' + name + '.csv', 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(np.array([np.mean(initial_losses),'avg']))
        writer.writerow(np.array([np.std(initial_losses),'std']))
    csvFile.close()

