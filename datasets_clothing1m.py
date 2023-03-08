import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utl
from PIL import Image
import random


NUM_CLASSES = {'mnist': 10, 'cifar10': 10, 'clothing1m': 14}


def baseline_labels(y_tr, dts, rds = 1):
    '''
    if the baseline labels exist, return the baseline labels.
    else, create it.
    '''
    data_baseline_file = "data/baseline_%s_%s_labels.npy" % (dts, rds) #this should be from kera, the old data
    if os.path.isfile(data_baseline_file):
        y_tr = np.load(data_baseline_file)
    else:
        np.random.seed(rds)
        for i in range(len(y_tr)):
            y_tr[i] = np.random.choice(NUM_CLASSES[dts])
        np.save(data_baseline_file, y_tr)
    return(data_baseline_file)

def new_lb(lb, nr, dts):
    pick = np.random.uniform()
    if pick <= nr/100.0:
        samples = list(range(NUM_CLASSES[dts]))
        samples.remove(lb)
        new_lb = np.random.choice(samples)
        return(new_lb)
    else:
        return(lb)

'''
def manipulate_labels(given_dataset_labels, dataset, c_class=0, noise_ratio=0, random_seed=1):
    y_train = np.load(given_dataset_labels)
    print('given dataset label shape is', y_train.shape)
    y_train = y_train.reshape([y_train.shape[0],])
    classes = list(range(NUM_CLASSES[dataset]))
    if noise_ratio[c_class] > 100:
        data_file = baseline_labels(y_train, dataset, random_seed)
    else:
        print('noise_ratio',noise_ratio)
        data_file = "data/%s_train_labels_seed_%s_add_%s.npy" % (dataset, random_seed, noise_ratio[0])#every seed changes 1 time labels
        if os.path.isfile(data_file): 
            y_train = np.load(data_file)
        else:
            np.random.seed(random_seed)
            for i in range(len(y_train)):
                y_train[i] = new_lb(y_train[i], noise_ratio[y_train[i]], dataset)
            np.save(data_file, y_train)
    return(data_file)
'''

class clothing_dataset(data_utl.Dataset): 
    def __init__(self, dir, noise_ratio, random_seed, transform, mode):
        self.train_imgs = []
        self.test_imgs = []

        self.test_labels = {}
        self.train_labels = []

        self.transform = transform
        self.mode = mode

        #with open('../images/clean_test_key_list.txt','r') as f:#clean_test_key_list.txt
        with open(dir+'/clean_test_key_list.txt','r') as f:#clean_test_key_list.txt
            lines = f.read().splitlines()
        for l in lines:
            #img_path = '../images/'+l[7:] #this replaces ./images/ to ./data/
            img_path = dir+'/'+l[7:] #this replaces ./images/ to ./data/
            self.test_imgs.append(img_path)
        
        #with open('../images/clean_label_kv.txt','r') as f:
        with open(dir+'/clean_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split() #eg, ['images/7/88/339645240,2782721788.jpg', '9']          
            #img_path = '../images/'+entry[0][7:] #this gives the image path, eg, './data/7/88/339645240,2782721788.jpg'
            img_path = dir+'/'+entry[0][7:] #this gives the image path, eg, './data/7/88/339645240,2782721788.jpg'
            self.test_labels[img_path] = int(entry[1])  
                   
        #with open('../images/matrix_subset_noisy_clean_subset_1.5k.txt','r') as f:
        #with open('../images/test_matrix_subset_noisy_clean_subset.txt','r') as f:
        #with open('../images/matrix_subset_noisy_clean.txt','r') as f:
        with open(dir+'/matrix_subset_noisy_clean.txt','r') as f:
            lines = f.read().splitlines()
       
        for l in lines:
            #self.train_imgs.append('../'+l.split()[0])
            self.train_imgs.append(dir+'/'+l[7:].split()[0])
            self.train_labels.append(int(l.split()[1]))#list version noisy set labels, if l.split()[2] then corresponding clean labels


        if noise_ratio[0] > 100:
            data_baseline_file = "./data/%s_%s_labels_%d.npy" % ('clothing1m', random_seed, noise_ratio[0]) 
            if os.path.isfile(data_baseline_file):
                self.train_labels = np.load(data_baseline_file)
            else:
                np.random.seed(random_seed)
                for k in range(len(self.train_labels)):
                    self.train_labels[k] = np.random.choice(14)
                np.save(data_baseline_file, self.train_labels)
        else:
            data_file = "./data/%s_train_labels_seed_%s_add_%s.npy" % ('clothing1m', random_seed, noise_ratio[0])#every seed changes 1 time labels
            if os.path.isfile(data_file):
                self.train_labels = np.load(data_file)

            else:
                np.random.seed(random_seed)
                for k in range(len(self.train_labels)):
                    self.train_labels[k] = new_lb(self.train_labels[k], noise_ratio[self.train_labels[k]], 'clothing1m')
                np.save(data_file, self.train_labels)

    
    def __getitem__(self, index):  
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            #if index < 2:
            #    print('train getitem', img_path,target)
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_imgs)
        elif self.mode=='test':
            return len(self.test_imgs)


class clothing_LID(data_utl.Dataset): 
    def __init__(self, dir, selected_indices, transform): 
        self.transform = transform
        self.LID_imgs = []
        #with open('../images/matrix_subset_noisy_clean.txt','r') as f:
        with open(dir+'/matrix_subset_noisy_clean.txt','r') as f:
            lines = f.read().splitlines()
        for l in selected_indices:
            #self.LID_imgs.append('../'+lines[l].split()[0])
            self.LID_imgs.append(dir+'/'+lines[l].split()[0][7:])

            
    def __getitem__(self, index):  
        img_path = self.LID_imgs[index]
        image = Image.open(img_path).convert('RGB')    
        img = self.transform(image)
        return img
    
    def __len__(self):
        return len(self.LID_imgs)