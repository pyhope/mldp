#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:30:38 2020

@author: jiedeng
"""

import numpy as np
import matplotlib.pyplot as plt
lcurve=np.loadtxt('lcurve.out')
if lcurve.shape[1]>6:
    print('deepmd-kit version<2')
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    ax[0][0].plot(lcurve[:,0], lcurve[:,1],'.',markersize=1.5,alpha=0.3, label= 'test')
    ax[0][0].plot(lcurve[:,0], lcurve[:,2],'*',markersize=1.5,alpha=0.3, label= 'train')
    ax[0][1].plot(lcurve[:,0], lcurve[:,3],'.',markersize=1.5,alpha=0.3,label= 'test')
    ax[0][1].plot(lcurve[:,0], lcurve[:,4],'*',markersize=1.5,alpha=0.3, label= 'train')
    ax[1][0].plot(lcurve[:,0], lcurve[:,5],'.',markersize=1.5,alpha=0.3, label= 'test')
    ax[1][0].plot(lcurve[:,0], lcurve[:,6],'*',markersize=1.5,alpha=0.3, label= 'train')
    if lcurve.shape[1]>8:
        ax[1][1].plot(lcurve[:,0], lcurve[:,7],'.',markersize=1.5,alpha=0.3, label= 'test')
        ax[1][1].plot(lcurve[:,0], lcurve[:,8],'*',markersize=1.5,alpha=0.3, label= 'train')
    ax[0][0].set_yscale('log')
    ax[0][1].set_yscale('log')
    ax[1][0].set_yscale('log')
    ax[1][1].set_yscale('log')
    
    ax[0][0].grid(True)
    ax[0][1].grid(True)
    ax[1][0].grid(True)
    ax[1][1].grid(True)
    
    ax[0][0].set_ylabel('tot')
    ax[0][1].set_ylabel('energy')
    ax[1][0].set_ylabel('force')
    ax[1][1].set_ylabel('viral')
    ax[0][0].legend()
    plt.show()
else: 
    print('deepmd-kit version>=2')
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    ax[0][0].plot(lcurve[:,0], lcurve[:,1],'.',markersize=1.5,alpha=0.3, label= 'train')
    ax[0][1].plot(lcurve[:,0], lcurve[:,2],'*',markersize=1.5,alpha=0.3, label= 'train')
    ax[1][0].plot(lcurve[:,0], lcurve[:,3],'.',markersize=1.5,alpha=0.3,label= 'train')
    ax[1][1].plot(lcurve[:,0], lcurve[:,4],'*',markersize=1.5,alpha=0.3, label= 'train')
    ax[0][0].set_yscale('log')
    ax[0][1].set_yscale('log')
    ax[1][0].set_yscale('log')
    ax[1][1].set_yscale('log')
    
    ax[0][0].grid(True)
    ax[0][1].grid(True)
    ax[1][0].grid(True)
    ax[1][1].grid(True)
    
    ax[0][0].set_ylabel('tot')
    ax[0][1].set_ylabel('energy')
    ax[1][0].set_ylabel('force')
    ax[1][1].set_ylabel('viral')
    ax[0][0].legend()
    plt.savefig('lcurve.png', dpi=300)
    plt.show()    

#    data = np.genfromtxt("lcurve.out", names=True) 
#    fig, ax = plt.subplots(2,2,figsize=(10,8))
#    for name in data.dtype.names[1:-1]:
#        plt.plot(data['step'], data[name], label=name) 
#    plt.legend()
#    plt.xlabel('Step') 
#    plt.ylabel('Loss') 
#    plt.xscale('symlog') 
#    plt.yscale('symlog') 
#    plt.grid() 
#    plt.show()