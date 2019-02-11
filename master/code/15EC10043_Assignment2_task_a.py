# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 18:53:56 2019

Name: Rajdeep Biswas
Roll No: 15EC10043
Assignment 2 Task a
"""
import argparse
from zipfile import ZipFile
import numpy as np
import mxnet as mx
from sklearn.model_selection import train_test_split
from mxnet import autograd, gluon,init
import mxnet.gluon.nn as mx_nn
import time
import matplotlib.pyplot as plt
import os
import fileloader

def acc(output, label):
    return 100*(output.argmax(axis=1) ==label.astype('float32')).mean().asscalar()

class DataLoader(object):
    def __init__(self):
        self.DIR = '../data/'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode='train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

def model_1_fit(no_epochs,batch_size,ctx,mx_train_data,mx_valid_data):
    train_loss_hist=[]
    train_acc_hist=[]
    valid_loss_hist=[]
    valid_acc_hist=[]
    
    mx_net = mx_nn.Sequential()
    mx_net.add(mx_nn.Dense(512, activation='relu'),
               mx_nn.Dense(128, activation='relu'),
               mx_nn.Dense(64, activation='relu'),
               mx_nn.Dense(32, activation='relu'),
               mx_nn.Dense(16, activation='relu'),
               mx_nn.Dense(10))
    print('The Model')
    print(mx_net)
    mx_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    #mx_net.initialize(init=init.Xavier())
    mx_net.collect_params().initialize(init.Uniform(), ctx=ctx)
    mx_trainer = gluon.Trainer(mx_net.collect_params(),'adam', {'learning_rate': 1e-3,'beta1':0.9,'beta2':0.999})
    print('\nFitting the model\n')
    for epoch in range(no_epochs):
        train_loss, train_acc, valid_acc,valid_loss = 0.0, 0.0, 0.0,0.0
        tic = time.time()
        for data, label in mx_train_data:
            with autograd.record():
                output=mx_net(data)
                loss = mx_loss_fn(output, label)
            loss.backward()
            mx_trainer.step(batch_size=batch_size)
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)
        for data, label in mx_valid_data:
            output=mx_net(data)
            loss = mx_loss_fn(output, label)
            valid_acc += acc(output, label)
            valid_loss += loss.mean().asscalar()
        train_loss_hist.append(train_loss/len(mx_train_data))
        train_acc_hist.append(train_acc/len(mx_train_data))
        valid_loss_hist.append(valid_loss/len(mx_valid_data))
        valid_acc_hist.append(valid_acc/len(mx_valid_data))
        
        print("Epoch %d: train loss %.3f, train acc %.3f %%, val loss %.3f, val acc %.3f %%, in %.1f sec" % (epoch, train_loss/len(mx_train_data), train_acc/len(mx_train_data),valid_loss/len(mx_valid_data),valid_acc/len(mx_valid_data), time.time()-tic))
        
    mx_net.save_parameters('../weights/task_a_model_1.params')
        
    
    return train_loss_hist,train_acc_hist,valid_loss_hist,valid_acc_hist
    
def model_2_fit(no_epochs,batch_size,ctx,mx_train_data,mx_valid_data):
    train_loss_hist=[]
    train_acc_hist=[]
    valid_loss_hist=[]
    valid_acc_hist=[]
    
    mx_net = mx_nn.Sequential()
    mx_net.add(mx_nn.Dense(1024, activation='relu'),
               mx_nn.Dense(512, activation='relu'),
               mx_nn.Dense(256, activation='relu'),
               mx_nn.Dense(10))
    print('The Model')
    print(mx_net)
    mx_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    #mx_net.initialize(init=init.Xavier())
    mx_net.collect_params().initialize(init.Uniform(), ctx=ctx)
    mx_trainer = gluon.Trainer(mx_net.collect_params(),'adam', {'learning_rate': 1e-3,'beta1':0.9,'beta2':0.999})
    print('\nFitting the model\n')
    for epoch in range(no_epochs):
        train_loss, train_acc, valid_acc,valid_loss = 0.0, 0.0, 0.0,0.0
        tic = time.time()
        for data, label in mx_train_data:
            with autograd.record():
                output=mx_net(data)
                loss = mx_loss_fn(output, label)
            loss.backward()
            mx_trainer.step(batch_size=batch_size)
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)
        for data, label in mx_valid_data:
            output=mx_net(data)
            loss = mx_loss_fn(output, label)
            valid_acc += acc(output, label)
            valid_loss += loss.mean().asscalar()
        train_loss_hist.append(train_loss/len(mx_train_data))
        train_acc_hist.append(train_acc/len(mx_train_data))
        valid_loss_hist.append(valid_loss/len(mx_valid_data))
        valid_acc_hist.append(valid_acc/len(mx_valid_data))
        
        print("Epoch %d: train loss %.3f, train acc %.3f %%, val loss %.3f, val acc %.3f %%, in %.1f sec" % (epoch, train_loss/len(mx_train_data), train_acc/len(mx_train_data),valid_loss/len(mx_valid_data),valid_acc/len(mx_valid_data), time.time()-tic))
        
    mx_net.save_parameters('../weights/task_a_model_2.params')
        
    
    return train_loss_hist,train_acc_hist,valid_loss_hist,valid_acc_hist    

def model_1_test(batch_size,ctx,mx_test_data):
    mx_net = mx_nn.Sequential()
    mx_net.add(mx_nn.Dense(512, activation='relu'),
               mx_nn.Dense(128, activation='relu'),
               mx_nn.Dense(64, activation='relu'),
               mx_nn.Dense(32, activation='relu'),
               mx_nn.Dense(16, activation='relu'),
               mx_nn.Dense(10))
    mx_net.load_parameters('../weights/task_a_model_1.params')
    mx_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    test_acc,test_loss=0.0,0.0
    for data, label in mx_test_data:
        output=mx_net(data)
        loss = mx_loss_fn(output, label)
        test_acc += acc(output, label)
        test_loss += loss.mean().asscalar()
    print('test loss %.3f, test acc %.3f %%'% (test_loss/len(mx_test_data), test_acc/len(mx_test_data)))    
    return

def model_2_test(batch_size,ctx,mx_test_data):
    mx_net = mx_nn.Sequential()
    mx_net.add(mx_nn.Dense(1024, activation='relu'),
               mx_nn.Dense(512, activation='relu'),
               mx_nn.Dense(256, activation='relu'),
               mx_nn.Dense(10))
    mx_net.load_parameters('../weights/task_a_model_2.params')
    mx_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    test_acc,test_loss=0.0,0.0
    for data, label in mx_test_data:
        output=mx_net(data)
        loss = mx_loss_fn(output, label)
        test_acc += acc(output, label)
        test_loss += loss.mean().asscalar()
    print('test loss %.3f, test acc %.3f %%'% (test_loss/len(mx_test_data), test_acc/len(mx_test_data)))    
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the models",action="store_true")
    parser.add_argument("--test", help="test the models",action="store_true")
    
    args = parser.parse_args()
    no_epochs = 40
    batch_size=256
    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    dataloader=DataLoader()
    if args.train:
        print('Training The Models...\n')  
        
        temp_data,temp_label=dataloader.load_data()
        temp_data=temp_data.astype(np.float32)
        
        train_data,valid_data,train_label,valid_label=train_test_split(temp_data,temp_label,test_size=0.3,random_state=42)
        im_train=train_data/255
        im_valid=valid_data/255
        train_dataset = mx.gluon.data.dataset.ArrayDataset(im_train, train_label)
        valid_dataset = mx.gluon.data.dataset.ArrayDataset(im_valid, valid_label)
        mx_train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        mx_valid_data = gluon.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        print('Traing the Model 1')
        
        train_loss_hist_m1,train_acc_hist_m1,valid_loss_hist_m1,valid_acc_hist_m1 =model_1_fit(no_epochs,batch_size,ctx,mx_train_data,mx_valid_data)
        print('Finished Traing the Model 1')
        print('Traing the Model 2')
        
        train_loss_hist_m2,train_acc_hist_m2,valid_loss_hist_m2,valid_acc_hist_m2 =model_2_fit(no_epochs,batch_size,ctx,mx_train_data,mx_valid_data)
        print('Finished Traing the Model 2')
        
        print('\nPlotting the Performance')
        epochs=np.arange(no_epochs)
        
        fig,ax = plt.subplots()
        ax.plot(epochs,train_loss_hist_m1,linewidth=2.0,label='Training Loss')
        ax.plot(epochs,valid_loss_hist_m1,linewidth=2.0,label='Validation Loss')
        ax.plot(epochs,train_loss_hist_m2,linewidth=2.0,label='Training Loss')
        ax.plot(epochs,valid_loss_hist_m2,linewidth=2.0,label='Validation Loss')
        h,l = ax.get_legend_handles_labels()
        ph1 = [plt.plot([],marker="",ls="")[0]]
        ph2 = [plt.plot([],marker="",ls="")[0]]
        handles = ph1+h[:2]+ph2+h[-2:]
        labels = ["$\it{Model}$ $\it{1}$"]+l[:2]+["$\it{Model}$ $\it{2}$"]+l[-2:]
        leg = plt.legend(handles,labels,ncol=2,bbox_to_anchor=(0.5,-0.2), loc=9,borderaxespad=0)
        for vpack in leg._legend_handle_box.get_children():
            for hpack in vpack.get_children()[:1]:
                hpack.get_children()[0].set_width(0)
        plt.grid(True)
        plt.xlabel('No of Epochs',fontsize='14',fontname ='Times New Roman')
        plt.ylabel('Cross Entropy Loss',fontsize='14',fontname ='Times New Roman')
        plt.title('Loss vs Epochs',fontsize='18',fontname ='Times New Roman')
        plt.savefig('task_a_loss.png', format='png', dpi=600,bbox_inches='tight')
        plt.close('all')
        
        fig,ax = plt.subplots()
        ax.plot(epochs,train_acc_hist_m1,linewidth=2.0,label='Training Accuracy')
        ax.plot(epochs,valid_acc_hist_m1,linewidth=2.0,label='Validation Accuracy')
        ax.plot(epochs,train_acc_hist_m2,linewidth=2.0,label='Training Accuracy')
        ax.plot(epochs,valid_acc_hist_m2,linewidth=2.0,label='Validation Accuracy')
        h,l = ax.get_legend_handles_labels()
        ph1 = [plt.plot([],marker="",ls="")[0]]
        ph2 = [plt.plot([],marker="",ls="")[0]]
        handles = ph1+h[:2]+ph2+h[-2:]
        labels = ["$\it{Model}$ $\it{1}$"]+l[:2]+["$\it{Model}$ $\it{2}$"]+l[-2:]
        leg = plt.legend(handles,labels,ncol=2,bbox_to_anchor=(0.5,-0.2), loc=9,borderaxespad=0)
        for vpack in leg._legend_handle_box.get_children():
            for hpack in vpack.get_children()[:1]:
                hpack.get_children()[0].set_width(0)
        plt.grid(True)
        plt.xlabel('No of Epochs',fontsize='14',fontname ='Times New Roman')
        plt.ylabel('Accuracy (%)',fontsize='14',fontname ='Times New Roman')
        plt.title('Accuracy vs Epochs',fontsize='18',fontname ='Times New Roman')
        plt.savefig('task_a_accuracy.png', format='png',dpi=600, bbox_inches='tight')
        plt.close('all')
        
    if args.test:
        print('\nTesting The Models...\n')
        test_data,test_label=dataloader.load_data(mode='test')
        test_data=test_data.astype(np.float32)
        im_test=test_data/255
        test_dataset = mx.gluon.data.dataset.ArrayDataset(im_test, test_label)
        
        mx_test_data = gluon.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print('\nTesting the Model 1')
        
        if os.path.isfile('../weights/task_a_model_1.params'):
            model_1_test(batch_size,ctx,mx_test_data)
        else:
            fileloader.download_file('task_a_model_1.params')
            model_1_test(batch_size,ctx,mx_test_data)
        
        print('\nTesting the Model 2')
        
        if os.path.isfile('../weights/task_a_model_2.params'):
            model_2_test(batch_size,ctx,mx_test_data)
        else:
            fileloader.download_file('task_a_model_2.params')
            model_2_test(batch_size,ctx,mx_test_data)

    