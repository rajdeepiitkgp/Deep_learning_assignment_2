# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 00:51:00 2019

Name: Rajdeep Biswas
Roll No: 15EC10043
Assignment 2 File Downloader

"""
import requests

def download_file(file_name):
    urla = 'https://github.com/rajdeepiitkgp/Deep_learning_assignment_2/blob/master/master/weights/'
    urlb= '?raw=true'
    url=urla+file_name+urlb
    proxies = {'http': 'http://172.16.2.30:8080','https': 'https://172.16.2.30:8080'}
    s = requests.Session()
    s.proxies = proxies
    
    r = requests.get(url)
    with open('../weights/'+file_name, 'wb') as out_file:
        out_file.write(r.content)