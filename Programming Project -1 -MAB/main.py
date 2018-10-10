# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:54:37 2018

@author: Administrator
"""

from greedyBandit import GreedyBandit 
#from optiBandit import run as runOpti
#from UCBIBandit import run as runUCBI
#from TSBandit import run as runTS
#from decGreedyBandit import run as runDecGreedy #(optional)

import matplotlib.pyplot as plt
 
times = 1000000 #Simulation times
greedy = GreedyBandit(0)
result = greedy.run(0.8,0.5,0.2,epsilon=0.1,N=times) 
plt.plot(result)
plt.show()
#runOpti(0.1,mu1=0.8,mu2=0.5,mu3=0.2)
#runUCBI(N)
#runTS(N)
#runDecGreedy(N)