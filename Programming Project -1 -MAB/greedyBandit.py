# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:52:33 2018

@author: Administrator
"""
import numpy as np

class GreedyBandit():
    def __init__(self,mu,mean):
        self.mu = mu
        self.mean = mean
        self.times = 0

    def pull(self):
        return np.random.randn()+self.mu #return a random reward value

    def update(self,xn):
        #Update predicted mean of bandit.
        self.times += 1
        self.mean = (self.mean*(self.times-1)+xn)/self.times

    def run(self,mu1,mu2,mu3,epsilon=0.1,N=10000):
        bandits = [GreedyBandit(mu1,self.mean),GreedyBandit(mu2,self.mean),GreedyBandit(mu3,self.mean)] #Declare bandits
        data = np.empty(N)          #data is used to store output of Epsilon Greedy Algorithm.
        
        for i in range(N): #Run Epsilon Greedy N times.
            p = np.random.random() #Get a random number p.
            
            if p < epsilon:
                #Randomly choose a bandit .
                j = np.random.choice(3) 
            else:
                #Choose bandit which has biggest mean.
                j = np.argmax([b.mean for b in bandits]) 
            
            x=bandits[j].pull()     #Pull the bandit and store the reward value.
            bandits[j].update(x)    #Update the mean value of bandit.
            data[i] = x             #Record the reward value.

        cumul_average = np.cumsum(data)/(np.arange(N)+1) #Calculate the cumulative reward.
 
        return cumul_average
    
        
    
        
    
    