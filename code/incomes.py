#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:02:03 2017

@author: Lauren
"""

import numpy as np
import matplotlib.pyplot as plt

#incomes=[]

incomes = np.random.exponential(2000,(1000,1))
    
#print(incomes)
    
plt.hist(incomes)