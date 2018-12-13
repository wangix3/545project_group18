# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:33:07 2018

@author: 25742
"""

#import optshrink as opt # package we create
import numpy as np
# import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
# Application 1: truncated svd vs optshrink algorithm
def Optshrink(Y,r):
    Y=np.mat(Y)
    U,s,V=np.linalg.svd(Y)
    m,n=Y.shape
    r=min(r,m,n)
    if m >= n:
        S=np.concatenate([np.diag(s[r:n]),np.zeros([(m-n),(n-r)])], axis=0)
    else:
        S=np.concatenate([np.diag(s[r:m]),np.zeros([(m-r),(n-m)])], axis=1)
    w=np.zeros(r)
    for k in range(0,r):
        D,Dder=D_transfrom_from_matrix(s[k],S)
        w[k]=-2*D/Dder
    Xh = U[:,0:r]*np.diag(w)*V[0:r:,]
    return Xh

def D_transfrom_from_matrix(z,X):
    X=np.mat(X)
    n,m=X.shape
    In=np.mat(np.diag(np.ones(n)))
    Im=np.mat(np.diag(np.ones(m)))
    D1=np.trace(z*(z*z*In-X*X.T)**(-1))/n
    D2=np.trace(z*(z*z*Im-X.T*X)**(-1))/m
    D=D1*D2
    D1_der=np.trace(-2*z*z*(z*z*In-X*X.T)**(-2)+(z*z*In-X*X.T)**(-1))/n
    D2_der=np.trace(-2*z*z*(z*z*Im-X.T*X)**(-2)+(z*z*Im-X.T*X)**(-1))/m
    D_der=D1*D2_der+D2*D1_der
    return D, D_der






# Application 2 : digit classification
# data exploration
def data_exploration(train_t):
    fig=plt.figure(figsize=(11,11))
    m,n,p=train_t.shape
    row=4
    column=5
    for i in range(1,row+1): 
        if  i%2 == 1:
            train=train_t+66*np.random.normal(np.zeros([m, n, p]))
            for j in range(0,column):
                digit=np.reshape(train[:,:,int(j+5*(i-1)/2)][:,1], [28,28])
                fig.add_subplot(row, column, 5*(i-1)+j+1, title='nosie plus digit ' + str(int(j+5*(i-1)/2)))
                plt.imshow(digit, cmap=plt.get_cmap('gray'))         
        if i%2 == 0:
            train=train_t
            for j in range(0,column):
                digit=np.reshape(train[:,:,int(j+5*(i-2)/2)][:,1], [28,28])
                fig.add_subplot(row, column, 5*(i-2)+j+1+5, title='original digit ' + str(int(j+5*(i-2)/2)))
                plt.imshow(digit, cmap=plt.get_cmap('gray'))
            

# function to change traindata (10,800,784) to (784,800,10)
def transformation_matrix(train):
    m,n,b=train.shape
    train_t=np.zeros([b,n,m])
    for i in range(0,m):
        for j in range(0,n):
            train_t[:,:,i][:,j]=train[i,:,:,][j,:]
    return train_t

# after model selection, we use the parameter r = 10. 
def nearest_ss_optshrink(train, ktrain): # ktrain = n
    n,N,d=train.shape
    U=np.zeros([n,ktrain,d])
    for j in range(0,d):
        Uj=np.linalg.svd(Optshrink(train[:,:,j],10))[0]
        U[:,:,j]=Uj[:,0:ktrain]
    return U

def nearest_ss(train, k): # ktrain = n
    n,N,d=train.shape
    U=np.zeros([n,k,d])
    for j in range(0,d):
        Uj=np.linalg.svd(train[:,:,j])[0]
        U[:,:,j]=Uj[:,0:k]
    return U

def classify_manylabels(test, U, k, test_label): 
    correst=0
    n,p=test.shape
    d=U.shape[2]
    err=np.zeros([d,p])
    for j in range(0,d):
        Uj=U[:,0:k-1,j]
        err[j,:]=np.sum(np.square(np.mat(test)-np.mat(Uj)*(np.mat(Uj).T*np.mat(test))),axis=0)
    label=np.argmin(err,axis=0)
    for i in range(0,len(label)):
        if label[i]==test_label[i]:
            correst+=1
    pcorrect=correst/len(label)
    return pcorrect

def accuary_plot(test,opt_U,orig_U,testlabel,maxN):
    pcorrect_opt=[]
    pcorrect_orig=[]
    for i in range(0,maxN):
        pcorrect_opt.append(classify_manylabels(test, opt_U, i+1, test_label))
    max_position=np.argmax(pcorrect_opt)
    max_value=pcorrect_opt[max_position]
    for i in range(0,maxN):
        pcorrect_orig.append(classify_manylabels(test, orig_U, i+1, test_label))
    plt.figure(figsize=(8,6))
    plt.plot(range(1,maxN+1), pcorrect_opt, c='blue', label='noise digit')
    plt.plot(range(1,maxN+1), pcorrect_orig, c='orange',linestyle='--', label='pure digit')
    plt.axvline(x=max_position+1, c='green', linestyle='--')
    plt.scatter([max_position+1], [max_value],c='red')
    plt.xlabel("Rank of the matrix")
    plt.ylabel("Accuracy %")
    plt.title('best classification accuracy = ' + str(max_value))
    plt.legend(loc='lower right', shadow=True)
    plt.show()
    return pcorrect_opt

def parameter_selection_opt(train_t, test_t, test_label, maxN, noise):
    m,n,p=train_t.shape
    train_n=train_t-noise*np.random.normal(np.zeros([m,n,p]))
    a,b=test_t.shape
    test_n=test_t-66*np.random.normal(np.zeros([a,b]))
    best_pcorrect=[]
    for r in range(1, 21):
        train_U_o=nearest_ss_optshrink(train_n, 10, r)
        pcorrect_opt=[]
        for i in range(0,maxN):
            pcorrect_opt.append(classify_manylabels(test_n, train_U_o, i+1, test_label))
        max_position_opt=np.argmax(pcorrect_opt)
        max_value_opt=pcorrect_opt[max_position_opt]
        best_pcorrect.append(max_value_opt)
    return best_pcorrect


    

    
    

# load data
traindata=h5py.File('train_digits.mat')
train=np.array(traindata["train_data"]) 
train_t=transformation_matrix(train)
testdata=h5py.File('test_digits.mat')
test_label=np.array(testdata["test_label"]) 
test_data=np.array(testdata["test_data"]) 
test_data=test_data.T 
# our pure data: train, test_data, test_label

# noise dataset:
np.random.seed(0)
m,n,p=train_t.shape
train_n=train_t-100*np.random.normal(np.zeros([m,n,p]))
a,b=test_data.shape
test_n=test_data-100*np.random.normal(np.zeros([a,b]))
