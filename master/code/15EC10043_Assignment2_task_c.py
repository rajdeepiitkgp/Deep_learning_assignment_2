# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:58:27 2019

Name: Rajdeep Biswas
Roll No: 15EC10043
Assignment 2 Task c
"""

import argparse
from zipfile import ZipFile
import numpy as np
import mxnet as mx
from sklearn.model_selection import train_test_split
from mxnet import autograd, gluon,init,nd
import time
import os
import fileloader
from sklearn.linear_model import LogisticRegression
import pickle

class MLP(gluon.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(1024)
            self.dense1 = gluon.nn.Dense(512)
            self.dense2 = gluon.nn.Dense(256)
            self.dense3 = gluon.nn.Dense(10)

    def forward(self, x):
        x1 = nd.relu(self.dense0(x))
        x2 = nd.relu(self.dense1(x1))
        x3 = nd.relu(self.dense2(x2))
        x4 = self.dense3(x3)
        return x1, x2, x3, x4


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
    
def model_fit(no_epochs,batch_size,ctx,mx_train_data,mx_valid_data):
    
    mx_net = MLP()

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
                x1,x2,x3,output=mx_net(data)
                loss = mx_loss_fn(output, label)
            loss.backward()
            mx_trainer.step(batch_size=batch_size)
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)
        for data, label in mx_valid_data:
            x1,x2,x3,output=mx_net(data)
            loss = mx_loss_fn(output, label)
            valid_acc += acc(output, label)
            valid_loss += loss.mean().asscalar()
        
        print("Epoch %d: train loss %.3f, train acc %.3f %%, val loss %.3f, val acc %.3f %%, in %.1f sec" % (epoch, train_loss/len(mx_train_data), train_acc/len(mx_train_data),valid_loss/len(mx_valid_data),valid_acc/len(mx_valid_data), time.time()-tic))
        
    mx_net.save_parameters('../weights/task_c_model_vanilla.params')        
    
    return mx_net

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the models",action="store_true")
    parser.add_argument("--test", help="test the models",action="store_true")
    batch_size=256
    args = parser.parse_args()
    no_epochs = 20
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
        
        mx_net = model_fit(no_epochs,batch_size,ctx,mx_train_data,mx_valid_data)
        mx_train_data = gluon.data.DataLoader(train_dataset, batch_size=len(train_data), shuffle=False)
        mx_valid_data = gluon.data.DataLoader(valid_dataset, batch_size=len(valid_data), shuffle=False)
        
        for data, label in mx_train_data:
            x1_train,x2_train,x3_train,output_train=mx_net(data)
            out_train=label
        
        for data, label in mx_valid_data:
            x1_valid,x2_valid,x3_valid,output_valid=mx_net(data)
            out_valid=label
            
        clf1 = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(x1_train.asnumpy(),out_train.asnumpy())
        clf2 = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(x2_train.asnumpy(),out_train.asnumpy())
        clf3 = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(x3_train.asnumpy(),out_train.asnumpy())
        
        filename1 = '../weights/task_c_model_1.sav'
        filename2 = '../weights/task_c_model_2.sav'
        filename3 = '../weights/task_c_model_3.sav'
        print('\nprinting scores')
        print('\nFor Model 1')
        print('Training Score %.3f Validation Score %.3f'%(clf1.score(x1_train.asnumpy(), out_train.asnumpy()),clf1.score(x1_valid.asnumpy(), out_valid.asnumpy())))
        print('\nFor Model 2')
        print('Training Score %.3f Validation Score %.3f'%(clf2.score(x2_train.asnumpy(), out_train.asnumpy()),clf2.score(x2_valid.asnumpy(), out_valid.asnumpy())))
        print('\nFor Model 3')
        print('Training Score %.3f Validation Score %.3f'%(clf3.score(x3_train.asnumpy(), out_train.asnumpy()),clf3.score(x3_valid.asnumpy(), out_valid.asnumpy())))
        
        pickle.dump(clf1, open(filename1, 'wb'))
        pickle.dump(clf2, open(filename2, 'wb'))
        pickle.dump(clf3, open(filename3, 'wb'))
        
    if args.test:
           
        print('\nTesting The Models...\n')
        test_data,test_label=dataloader.load_data(mode='test')
        test_data=test_data.astype(np.float32)
        im_test=test_data/255
        test_dataset = mx.gluon.data.dataset.ArrayDataset(im_test, test_label)
        
        mx_test_data = gluon.data.DataLoader(test_dataset, batch_size=len(test_data), shuffle=False)
        print('\nTesting the Model 1 with vanilla network')
        if not os.path.isfile('../weights/task_c_model_vanilla.params'):
            fileloader.download_file('task_c_model_vanilla.params')
        if not os.path.isfile('../weights/task_c_model_1.sav'):
            fileloader.download_file('task_c_model_1.sav')
        if not os.path.isfile('../weights/task_c_model_2.sav'):
            fileloader.download_file('task_c_model_2.sav')
        if not os.path.isfile('../weights/task_c_model_3.sav'):
            fileloader.download_file('task_c_model_3.sav')
        print('Vanilla Model Loss')
        mx_net=MLP()
        mx_net.load_parameters('../weights/task_c_model_vanilla.params')
        mx_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        test_acc,test_loss=0.0,0.0
        for data, label in mx_test_data:
            x1_test,x2_test,x3_test,output=mx_net(data)
            loss = mx_loss_fn(output, label)
            test_acc += acc(output, label)
            test_loss += loss.mean().asscalar()
            out_test=label
        print('test loss %.3f, test acc %.3f %%'% (test_loss/len(mx_test_data), test_acc/len(mx_test_data))) 
        clf1=pickle.load(open('../weights/task_c_model_1.sav', 'rb'))
        clf2=pickle.load(open('../weights/task_c_model_2.sav', 'rb'))
        clf3=pickle.load(open('../weights/task_c_model_3.sav', 'rb'))
        
        print('Logistic Model 1 Accuracy %.3f %%'%(clf1.score(x1_test.asnumpy(), out_test.asnumpy())*100))
        print('Logistic Model 2 Accuracy %.3f %%'%(clf2.score(x2_test.asnumpy(), out_test.asnumpy())*100))
        print('Logistic Model 3 Accuracy %.3f %%'%(clf3.score(x3_test.asnumpy(), out_test.asnumpy())*100))
        
        
        
        
