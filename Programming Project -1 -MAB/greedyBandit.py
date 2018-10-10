# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:52:33 2018

@author: Administrator
"""
import numpy as np

class GreedyBandit():
    
    def __init__(self,mu):
        self.mu = mu
        self.mean = 0.0
        self.times = 0
    def run(self,mu1,mu2,mu3,epsilon=0.1,N=10000):
        #Declare bandits
        bandits = [GreedyBandit(mu1),GreedyBandit(mu2),GreedyBandit(mu3)]

        #data is used to store output of Epsilon Greedy Algorithm.
        data = np.empty(N)
        
        #Run Epsilon Greedy N times.
        for i in range(N):
        
            #Randomly select a "p".
            p = np.random.random()
            if p < epsilon:
                #Randomly choose a bandit .
                j = np.random.choice(3)
            else:
                #Choose bandit which has biggest mean.
                j = np.argmax([b.mean for b in bandits])
            
            #Pull the bandit and store the reward value.
            x=bandits[j].pull()

            #Update the mean value of bandit.
            bandits[j].update(x)
            
            #Record the value every time.
            data[i] = x
        
        
        cumul_average = np.cumsum(data)/(np.arange(N)+1)

        print(N)
        for i in range(3):
            print("bandit"+str(i+1)+".mean="+str(bandits[i].mean))
        return cumul_average
    def pull(self):
        #return a random reward value
        return np.random.randn()+self.mu
    def update(self,xn):
        #Update predicted mean of bandit.
        self.times += 1
        self.mean = (self.mean*(self.times-1)+xn)/self.times
        #self.mean = (1-1.0/self.times)*self.mean + 1.0/self.times * xn
        
    
        
    
    