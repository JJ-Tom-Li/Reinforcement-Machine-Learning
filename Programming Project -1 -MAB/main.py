# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:54:37 2018

@author: Administrator
"""

from greedyBandit import GreedyBandit 
from decGreedyBandit import DecGreedyBandit
from optiBandit import OptiBandit
from UCBBandit import UCBBandit
from TSBandit import TSBandit


import matplotlib.pyplot as plt
 
times = 10000 #Simulation times

greedy = GreedyBandit(0,0)        #Greedy Bandit
dec_greedy = DecGreedyBandit(0) #Decreasing Greedy Bandit
opti = OptiBandit(0)            #Optimistic Initial Value Bandit
ucb = UCBBandit(0)              #UCB Bandit
ts = TSBandit(0)                #TS Bandit

#Function Arguments
m1=0.9
m2=0.5
m3=0.2
e=0.1

#Run Epsilon Greedy 
greedy_result = greedy.run(m1,m2,m3,e,times) 

#Run Decreasing Epsilon Greedy 
dec_greedy_result = dec_greedy.run(m1,m2,m3,times)

#Run Optimistic Initial Value 
opti_result = opti.run(m1,m2,m3,e,times)

#Run UCB 
ucb_result = ucb.run(m1,m2,m3,times) 

#Run Thompson Sampling 
ts_result = ts.run(m1,m2,m3,times) 

#Plot the result
plt.plot(greedy_result,label="Epsilon Greedy")
plt.plot(dec_greedy_result,label="Decreasing Epsilon Greedy")
plt.plot(opti_result,label="Optimistic Initial Value")
plt.plot(ucb_result,label="UCB1")
plt.plot(ts_result,label="Thompson Sampling")

#Show the graph
plt.legend(loc='upper right')
plt.show()