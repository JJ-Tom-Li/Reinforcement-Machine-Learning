# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:59:56 2018

@author: Administrator
"""
import numpy as np
class TSBandit():
    def __init__(self,mu,win=1,loss=1,mean=0):
        self.mu=mu              #the true mean value
        self.win=win            #the times of wins
        self.loss=loss          #the times of losses
        self.mean=mean          #the value drawn from beta distribution
        
    def reward(self):           #the reward for playing
        if(self.mu>np.random.random()):   #the player wins this round
            self.win+=1
            return 1
        self.loss+=1     #the player loses this round
        return 0
    
    def update(self):     #update information when being played
        self.mean=np.random.beta(self.win,self.loss)     #update beta distribution for this machine

    def run(self,mu1,mu2,mu3,N=10000):
        bandits=[TSBandit(mu1),TSBandit(mu2),TSBandit(mu3)] #3 different machines
        data=np.empty(N)
        for i in range(N):   #play N times
            chosen=np.argmax([b.mean for b in bandits]) #choose the machine according to Thompson sampling
            r=bandits[chosen].reward()
            bandits[chosen].update() 
            data[i]=r
        cumul_average=np.cumsum(data)/(np.arange(N)+1)   #counting the cumulative average

        return cumul_average