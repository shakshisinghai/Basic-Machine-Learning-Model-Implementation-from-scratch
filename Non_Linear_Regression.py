# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 06:24:15 2020

@author: ishus
"""
from random import random
import numpy as np
import matplotlib.pyplot as plt

    
#dataset
dataset = np.array([[i, i**2 ] for i in [random()*100 for i in range(50)]])
dataset = np.array([[i, i/2] for i in [random()*100 for i in range(50)]])




degree=2

#Learning rate
alpha=1
q=np.array([random() for i in range(degree)])
error_list=[]
n_epoch = 25000

while True:
    featuresX=(np.array([dataset[:,0]**i for i in range (degree)])).transpose()
    
    #normalization
    xmin=featuresX[:,1:].min(axis=0)
    xmax=featuresX[:,1:].max(axis=0)
    featuresX= np.append(np.expand_dims(featuresX[:,0],1),(featuresX[:,1:]-xmin)/(xmax-xmin),axis=1)
     
    
    hypothesis=featuresX.dot(q)
    
    Gradient=(hypothesis-dataset[:,1])*alpha/len(dataset)
    
    Gradient=(Gradient.dot(featuresX))/len(dataset)

    q=q-Gradient
    
    error_list.append(sum((hypothesis-dataset[:,1])**2)/len(dataset))
    print ("Error: %.4f, Total_epoch: %d" %(error_list[-1],len(error_list))) 

    if len(error_list)>1 and (error_list[-2]-error_list[-1])<(10**(-4)) :
        break

print ("Error: %.4f, Total_epoch: %d" %(error_list[-1],len(error_list))) 
plt.plot(featuresX[:,1], dataset[:,1], 'ro')
x=np.linspace(0,10, 100)
y=0
for i in range(degree):
    y=y+q[i]*x**i
plt.plot(x,y,'b-')
plt.xlim([min(featuresX[:,1]),max(featuresX[:,1])])
plt.ylim([min(dataset[:,1]),max(dataset[:,1])])

#plt.xticks(np.arange(0,max(dataset[:,0]),10))
#plt.yticks(np.arange(0,max(dataset[:,1]),10))
plt.show()


plt.plot(range(len(error_list)), error_list)
plt.show()

