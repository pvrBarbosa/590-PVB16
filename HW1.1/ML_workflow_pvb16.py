# -*- coding: utf-8 -*-
"""
@author: pvb16
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from scipy.optimize import minimize




class Data:
    
    def __init__(self):
        self._read_data()

        
        
    def _read_data(self):
        # Opening JSON file
        f = open('weight.json')
        # store JSON object as a dictionary
        self.data = json.load(f)
        # store the data as Pandas dataframe
        self.df = pd.DataFrame(self.data)
        
    def visualize(self, xval, yval, xlabel, ylabel):
        # Initialize the figure
        plt.figure()
        FS=18
        # Set the label names
        plt.xlabel(xlabel, fontsize=FS)
        plt.ylabel(ylabel, fontsize=FS)
        plt.title("Data visualizaton", fontsize=FS*1.2)
        # Plot
        plt.plot(xval,yval,'bo')
        plt.show()
                

##############################################################################
##################      FUNCTIONS AREA      ##################################
##############################################################################

def calc_mse(vector_a, vector_b):
    # Calculate the MSE
    n = len(vector_a)
    se = 0
    for i in range(n):
        se = se + (vector_a[i] - vector_b[i])**2
    mse = se / n
    return mse

def calc_mae(vector_a, vector_b):
    # Calculate the MAE
    n = len(vector_a)
    ae = 0
    for i in range(n):
        ae = ae + abs((vector_a[i] - vector_b[i]))
    mae = ae / n
    return mae
    
    
def scale(data):
    # Use standard scale to normalize the data
    mu = np.mean(data)
    stdev = np.std(data)
    scaled_data = []
    for i in range(len(data)):
        scaled_data.append((data[i] - mu)/stdev)
    return np.array(scaled_data)

def unscale(scaled_data, original_data):
    # Undo the data normalization
    mu = np.mean(original_data)
    stdev = np.std(original_data)
    unscaled_data = []
    for i in range(len(scaled_data)):
        unscaled_data.append(scaled_data[i] * stdev + mu)
    return np.array(unscaled_data)
    
          
    
def model(x, p):
    # Define models equations
    if(model_type=="linear"): 
        NFIT = 2
        return p[0]*x+p[1]  
    if(model_type=="logistic"): 
        NFIT = 4
        return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))


def loss(p):
    global iteration,iterations,loss_train,loss_test
    
    # Calculate train loss
    yp = model(s_x_train, p)
    train_loss = calc_mse(s_y_train, yp)
    
    # Calculate test loss
    yp_test = model(s_x_test, p)
    test_loss = calc_mse(s_y_test, yp_test)
    
    # Append to vector to plot later
    loss_train.append(train_loss)
    loss_test.append(test_loss)
    iterations.append(iteration)

    iteration+=1
    
    return train_loss
    

def train_model(m_type = "linear"):   

    # Initialize variables to plot
    global iteration,iterations,loss_train,loss_test, model_type
    iterations=[]; loss_train=[];  loss_test=[]
    iteration=0     
    
    
    model_type = m_type
    if(model_type=="linear"): 
        NFIT = 2 
    if(model_type=="logistic"): 
        NFIT = 4
    #RANDOM INITIAL GUESS FOR FITTING PARAMETERS
    po=np.random.uniform(0.5,1.,size=NFIT)
    #TRAIN MODEL USING SCIPY OPTIMIZER
    res = minimize(loss, po, method='Nelder-Mead', tol=1e-15)
    popt=res.x
    print("OPTIMAL PARAM:",popt)
    return popt



def plot_model(title, xlab = "age", ylab = "weight"):
    
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'o', label='Training set')
    ax.plot(x_test, y_test, 'x', label='Test set')
    ax.plot(np.sort(x_train), np.sort(y_pred), 'r-', label="Model")
    
    ax.legend()
    FS=18   #FONT SIZE
    plt.xlabel(xlab, fontsize=FS)
    plt.ylabel(ylab, fontsize=FS)
    plt.title(title, fontsize=FS*1.2)
    plt.show()
    
    
def plot_opt(title):
    
    fig, ax = plt.subplots()
    ax.plot(iterations, loss_train, 'o', label='Train loss')
    ax.plot(iterations, loss_test, 'o', label='Test loss')
    
    ax.legend()
    FS=18   #FONT SIZE
    plt.xlabel('optimizer iterations', fontsize=FS)
    plt.ylabel('loss', fontsize=FS)
    plt.title(title, fontsize=FS*1.2)
    
    plt.show()
        
        
        
##############################################################################
#######################     INITIALIZATION     ###############################
##############################################################################

# Create the class
weight = Data()

# Check the data top rows
weight.df.head()

# Visualize the data
weight.visualize(xval = weight.df["x"], 
                 yval = weight.df["y"], 
                 xlabel = weight.df["xlabel"][0], 
                 ylabel = weight.df["ylabel"][0])



##############################################################################
###################       LINEAR REGRESSION         ##########################
##############################################################################


# Define x and y for ages < 18
df18 = weight.df.loc[weight.df['x'] < 18]
x = df18['x']
y = df18['y']

# Train test split
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.20, random_state=0)

# Convert the variables to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Scale the variables
s_x_train = scale(x_train)
s_y_train = scale(y_train)
s_x_test = scale(x_test)
s_y_test = scale(y_test)

# Fit the model
popt = train_model("linear")

# Unscale the prediction
y_pred = unscale(model(s_x_train, popt), y_train)


# Plot the optimizations steps
plot_opt("Linear Regression Optimization")

# Plot the model
plot_model("Linear Regression Model")




##############################################################################
###################   LOGISTIC REGRESSION (WEIGHT)   ########################
##############################################################################


# Define x and y
x = weight.df['x']
y = weight.df['y']

# Train test split
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.20, random_state=0)

# Convert the variables to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Scale the variables
s_x_train = scale(x_train)
s_y_train = scale(y_train)
s_x_test = scale(x_test)
s_y_test = scale(y_test)

# Fit the model
popt = train_model("logistic")

# Unscale the prediction
y_pred = unscale(model(s_x_train, popt), y_train)


# Plot the optimizations steps
plot_opt("Logistic Regression Optimization")

# Plot the model
plot_model("Logistic Regression Model")


##############################################################################
###################   LOGISTIC REGRESSION (ADULT?)   ########################
##############################################################################


# Define x and y
x = weight.df['y']
y = weight.df['is_adult']

# Train test split
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.20, random_state=0)

# Convert the variables to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Scale the variables
s_x_train = scale(x_train)
s_y_train = scale(y_train)
s_x_test = scale(x_test)
s_y_test = scale(y_test)

# Fit the model
popt = train_model("logistic")

# Unscale the prediction
y_pred = unscale(model(s_x_train, popt), y_train)


# Plot the optimizations steps
plot_opt("Logistic Regression Optimization")

# Plot the model
plot_model("Logistic Regression Model", xlab = "weight", ylab = "is_adult")

















