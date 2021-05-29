import numpy as np
import argparse
import csv
import scipy.stats
import math
import os
import torch
from scipy.spatial.distance import pdist, cdist, squareform
import random

def fwd_KL(P, Q, cr, fill_zero=True):
    P = np.asarray(P) #the ground_truth matrix
    Q = np.asarray(Q) #the estimate
    Q /= Q.sum(axis=1, keepdims=True) #incase the estimated matrix is not row normalized
    assert P.shape[0] == P.shape[1]
    assert Q.shape[0] == Q.shape[1]
    assert P.shape[0] == Q.shape[1]
    assert P.shape[0] == len(cr)
    
    #if the estimate has 0 values, KL loss by default is infinity
    #therefore, substitue 0 with a small number
    if fill_zero==True:
        Q = np.where(Q == 0., 1.e-8, Q)
    l = P*np.log2(P/Q) #base 2
    l=np.where(np.isnan(l),0.,l)# loss for entries in P with value 0 will be nan, need to replace it.
    l = l.sum(axis = 1)
    cr = np.asarray(cr)
    loss = np.dot(cr, l)

    return(loss)

def sym_C(mixing_ratio, num_classes):
    m = np.full((num_classes, num_classes), mixing_ratio / (num_classes-1))
    for i in range(num_classes):
        m[i,i] = 1.-mixing_ratio
    #print('sym matrix', m)
    return m

def pw_C(mixing_ratio, num_classes):
    m = np.eye(num_classes)*(1.-mixing_ratio)
    for i in range(num_classes-1):
        m[i,i+1] = mixing_ratio
    m[num_classes-1,0] = mixing_ratio
    #print('pw matrix', m)
    return m

def true_p(form, mixing_ratio, num_classes=10):
    if form == 'sym':
        m = sym_C(mixing_ratio, num_classes)
        return m
    elif form == 'pw':
        m = pw_C(mixing_ratio, num_classes)
        return m
    else:
        assert 'noise type is not implemented!'


def noisify(loc, fm, r, dts = 'cifar10', save_path = './given_labels', random_seed=42):

    y_tr = np.load(loc)
    num_class = len(np.unique(y_tr))

    P = true_p(fm, r)

    np.random.seed(random_seed)
    for i in range(len(y_tr)):#noisify data according to the true p
        y_tr[i] = np.random.choice(num_class,p=P[y_tr[i]])
    return(y_tr)





def Q_estima(e_p, e_pp, alpha_vector, p_matrix, k):
    Q = np.zeros((k,k))
    prior_matrix = p_matrix.copy()
    prior_matrix /= prior_matrix.sum(axis=1, keepdims=True)#in case it is not row normalized
    for i in range(k):
        col_index = np.argmax(prior_matrix[i,:]) #get the max one
        prior_matrix[i,:] = prior_matrix[i,:]/(1-prior_matrix[i,col_index]) #normalize the prior without the max entry
        common_term = 0
        for t in range(k):
            if t != col_index:
                common_term += alpha_vector[t]*(1+e_pp)*prior_matrix[i,t]/(float(k-1))
        #estimate the max value for the ith row
        q = ((1-e_p)/float(k) - common_term)/(1-alpha_vector[col_index] - common_term)
        #fill the remaining entries according to the ratios
        Q[i,:]= prior_matrix[i,:]*(1.0-q)
        Q[i,col_index] = q
    return(Q)

def alpha_v(alpha_best, k, initial=True):
    if initial:
        alpha_vector = np.ones((k))*alpha_best
    else:
        alpha_vector = base_alpha_vector.copy()
        alpha_vector[tuning_c] = alpha_best #change the tuning class's alpha only
    alpha_vector /= 100.
    return(alpha_vector)

def epsilons(recall, k, N=1280):
    
    if float(recall) -1 == 0:
        z = -4.2
    else:
        z = scipy.stats.norm.ppf((1.0-float(recall))/2.0)
    e_p = z*math.sqrt((k-1)/N) #e of baseline
    e_pp = -e_p*(k-2)/float(k-1) #e of Q_alpha_best
    return(e_p, e_pp)


def make_determine(random_seed=0):
    random.seed(random_seed) #anything need to use random will be fixed
    os.environ['PYTHONHASHSEED'] = str(random_seed) #https://docs.python.org/3.3/using/cmdline.html
    np.random.seed(random_seed)
    torch.manual_seed(random_seed) 
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    def _init_fn():
        np.random.seed(random_seed)

def mle_batch(data, batch, k=20):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / (v[-1] + 1e-8)))
    a = cdist(batch, data)
    #b = np.apply_along_axis(np.sort, axis=1, arr=a)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def accuracy(output, target, topk=(1,1,)): 
    """Computes top 1 accuracy"""
    correct = 0.0
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = torch.squeeze(pred)
    correct = (pred.eq(target)).float().sum(0)
    return(correct.item())