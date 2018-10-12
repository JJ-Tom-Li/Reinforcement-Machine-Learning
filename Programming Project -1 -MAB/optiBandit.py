# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:53:06 2018

@author: Administrator
"""
import numpy as np
from greedyBandit import GreedyBandit 
class OptiBandit():
    def __init__(self,mu):
        self.mu = mu
        self.mean = 1.0     #提高初始mean值，使explore機率提高
        self.times = 0

    def pull(self):
        return np.random.randn()+self.mu #return a random reward value

    def update(self,xn):
        #Update predicted mean of bandit.
        self.times += 1
        self.mean = (self.mean*(self.times-1)+xn)/self.times

    def run(self,mu1,mu2,mu3,epsilon=0.1,N=10000):
        #Run Epsilon Greedy
        gb = GreedyBandit(0,self.mean)  
        cumul_average = gb.run(mu1,mu2,mu3,epsilon,N)
        return cumul_average
    
    