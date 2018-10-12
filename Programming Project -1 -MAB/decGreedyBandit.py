# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:52:33 2018

@author: Administrator
"""
import numpy as np

class DecGreedyBandit():
    def __init__(self,mu):
        self.mu = mu
        self.mean = 0.0
        self.times = 0

    def pull(self):
        return np.random.randn()+self.mu #return a random reward value

    def update(self,xn):
        #Update predicted mean of bandit.
        self.times += 1
        self.mean = (self.mean*(self.times-1)+xn)/self.times

    def run(self,mu1,mu2,mu3,N=10000):
        bandits = [DecGreedyBandit(mu1),DecGreedyBandit(mu2),DecGreedyBandit(mu3)] #Declare bandits
        data = np.empty(N) #data is used to store output of Epsilon Greedy Algorithm.

        #Run Decreasing Epsilon Greedy N times.
        for i in range(N):
            epsilon = (1 / (i*0.1+1)) #Decrease the epsilon.

            p = np.random.random()
            if p < epsilon:
                #Randomly choose a bandit .
                j = np.random.choice(3)
            else:
                #Choose bandit which has highest mean.
                j = np.argmax([b.mean for b in bandits])
            x=bandits[j].pull()     #Pull the bandit and store the reward value.
            bandits[j].update(x)    #Update the mean value of bandit.
            data[i] = x             #Record the value every time.
        
        cumul_average = np.cumsum(data)/(np.arange(N)+1) #Calculate the cumulative reward.

        return cumul_average
    
        
    
        
    
    