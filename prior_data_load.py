#modified from https://github.com/xiaoboxia/T-Revision

import numpy as np
import torch.utils.data as Data
from PIL import Image
import os

class cifar10_dataset(Data.Dataset):
    def __init__(self, temp_dir, name, data_dir, label_dir, train, transform=None, target_transform=None, random_seed=1, split_per=0.9, num_class=10):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        given_images = np.load(data_dir)
        given_labels = np.load(label_dir)

        num_samples = int(len(given_labels))

        np.random.seed(random_seed)
        train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
        index = np.arange(len(given_images))
        val_set_index = np.delete(index, train_set_index)
        self.train_data, self.val_data = given_images[train_set_index, :], given_images[val_set_index, :]
        self.train_labels, self.val_labels = given_labels[train_set_index], given_labels[val_set_index]

        if self.train:
            np.save(temp_dir+'/90pcnoisytrainlb_'+name+'.npy', self.train_labels)      
            self.train_data = self.train_data.reshape((len(self.train_data),3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        
        else:
            self.val_data = self.val_data.reshape((len(self.val_data),3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)

class cifar10_train(Data.Dataset):

    def __init__(self,train_img, noisy_lb, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train_data = np.load(train_img)
        self.train_data = self.train_data.reshape((len(self.train_data),3,32,32))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))

        self.train_labels = np.load(noisy_lb)

    def __getitem__(self, index):
        img, label = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)    
     
        return(img, label)


    def __len__(self):
        return len(self.train_data)

class cifar10_test(Data.Dataset):
    def __init__(self, data_path, label_path, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
        self.val_data = np.load(data_path)
        self.val_data = self.val_data.reshape((len(self.val_data),3,32,32))
        self.val_data = self.val_data.transpose((0, 2, 3, 1))
        self.val_labels = np.load(label_path)
        self.val_labels = self.val_labels.reshape([self.val_labels.shape[0],])

    def __getitem__(self, index):
        
        img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)    
     
        return(img, label)

    def __len__(self):

        return(len(self.val_data))

class cifar10_LID(Data.Dataset):

    def __init__(self, train_img, selected_indices, transform=None, num_class=10):
    
        self.transform = transform        
        self.train_data = np.load(train_img)
        self.train_data = self.train_data[selected_indices]
        self.train_data = self.train_data.reshape((len(self.train_data),3,32,32))
        self.train_data = self.train_data.transpose((0, 2, 3, 1)) 

    def __getitem__(self, index):
           
        img = self.train_data[index]
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
     
        return(img)

    def __len__(self):
            
        return(len(self.train_data))

def baseline_labels(y_tr, dts, rds = 1, num_classes=10):
    '''
    if the baseline labels exist, return the baseline labels.
    else, create it.
    '''
    data_baseline_file = "data/baseline_%s_%s_labels.npy" % (dts, rds) 
    if os.path.isfile(data_baseline_file):
        y_tr = np.load(data_baseline_file)
    else:
        np.random.seed(rds)
        for i in range(len(y_tr)):
            y_tr[i] = np.random.choice(num_classes)
        np.save(data_baseline_file, y_tr)
    return(data_baseline_file)

def new_lb(lb, nr, dts, num_classes=10):
    pick = np.random.uniform()
    if pick <= nr/100.0:
        samples = list(range(num_classes))
        samples.remove(lb)
        new_lb = np.random.choice(samples)
        return(new_lb)
    else:
        return(lb)

def manipulate_labels(given_dataset_labels, dataset, noise_ratio, c_class=0, random_seed=1, num_classes=10):
    y_train = np.load(given_dataset_labels)
    y_train = y_train.reshape([y_train.shape[0],])
    classes = list(range(num_classes))
    if noise_ratio[c_class] > 100:
        data_file = baseline_labels(y_train, dataset, random_seed)
    else:
        print('noise_ratio',noise_ratio)
        data_file = "data/%s_train_labels_seed_%s_add_c%d_%s.npy" % (dataset, random_seed, c_class, noise_ratio[c_class])#every seed changes 1 time labels
        if os.path.isfile(data_file): 
            y_train = np.load(data_file)
        else:
            np.random.seed(random_seed)
            for i in range(len(y_train)):
                y_train[i] = new_lb(y_train[i], noise_ratio[y_train[i]], dataset)
            np.save(data_file, y_train)
    return(data_file)

