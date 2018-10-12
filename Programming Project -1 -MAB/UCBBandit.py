# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:54:16 2018

@author: Administrator
"""
import numpy as np
import math

class UCBBandit():
    def __init__(self,mu,mean=0,ucb=0,times=0):
        self.mu=mu          #the true mean value
        self.mean=mean      #assumed mean
        self.ucb=ucb        #upper confidence bound
        self.times=times    #times of being played
        
    def reward(self):                       #the reward for playing
        return self.mu+np.random.randn()
        
    def update(self,reward,total_times):       #update when being played
        self.mean=((self.mean*self.times)+reward)/(self.times+1)    #update mean
        self.times+=1                                                      #update times of being played
        self.ucb=self.mean+math.sqrt(2*math.log(total_times/self.times))   #update upper confidence bound
    
    def run(self,mu1,mu2,mu3,N=10000):
        bandits=[UCBBandit(mu1),UCBBandit(mu2),UCBBandit(mu3)] #3 different machines
        data=np.empty(N)
        for i in range(N):   #play N times
            chosen=np.argmax([b.ucb for b in bandits]) #choose the machine with the highest upper confidence bound
            r=bandits[chosen].reward()
            bandits[chosen].update(r,N)
            data[i]=r
        cumul_average=np.cumsum(data)/(np.arange(N)+1)   #counting the cumulative average

        
        return cumul_average