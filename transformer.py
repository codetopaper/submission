
import torch
import numpy as np
import torchvision.transforms as transforms

def transform_train(dataset_name):
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
   ])
    
    return transform


def transform_test(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
   ])
    
    return transform


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    


def no_rc_transform_train(dataset_name='cifar10'):
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
   ])
    
    return transform