#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 17:50:05 2021

@author: pedrob
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json


INPUT_FILE = "planar_x1_x2_x3_y.json"


# Read in the file
with open(INPUT_FILE) as f:
	my_input = json.load(f)

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
data=[];
for key in my_input.keys():
	data.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
data=np.transpose(np.array(data))

#SELECT COLUMNS FOR TRAINING 
X=data[:,0:-1];  Y=data[:,-1]
Y = np.reshape(Y,(len(Y),1))
