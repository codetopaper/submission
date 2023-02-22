import numpy as np
import torch.utils.data as Data
from PIL import Image




class clothing1m_dataset(Data.Dataset): 
    def __init__(self, data_file, name, train=True, transform=None, target_transform=None,split_per=0.9,random_seed=0): 
        self.train_imgs = []
        self.val_imgs = []
        self.train_labels = []
        self.val_labels = []

        self.transform = transform
        self.target_transform = target_transform
        
        self.train = train

        path = data_file.split('images')[0]

        with open(data_file,'r') as f:
            lines = f.read().splitlines()
        
        np.random.seed(random_seed)
        for l in lines:
            pick = np.random.uniform()
            im, n_lb, c_lb = l.split()
            if pick <= split_per:
                self.train_imgs.append(path+im)
                self.train_labels.append(int(n_lb))#list version noisy set labels
            else:
                self.val_imgs.append(path+im)
                self.val_labels.append(int(n_lb))#list version noisy set labels
                
        np.save('90pctrainlb_'+name+'.npy', self.train_labels)

    def __getitem__(self, index):
        if self.train:
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
        else:
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        if self.train:
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)




