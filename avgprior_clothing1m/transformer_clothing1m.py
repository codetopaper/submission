import torch
import numpy as np
import torchvision.transforms as transforms


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    


def clothing_tran(mode):
    if mode == "train":
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) # meanstd transformation
    else:
        transform = transforms.Compose([
                transforms.Resize(256),
                #transforms.RandomSizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) # meanstd transformation
    return transform