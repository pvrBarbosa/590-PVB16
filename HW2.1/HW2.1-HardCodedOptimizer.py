import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

"""
###############################################################################
To make the testing easier change the optimization parameters here
###############################################################################
"""
method = "batch"    # options: "batch", "mini-batch", "stochastic"
algo = "GD"         # "GDM" for Gradient Descent with Momentum
LR = 0.0001         # Learning rate
n_batches = 2       # Number of batches in case of mini-batch method.
                    # If n_batches = 2 then batch size = 0.5
decay_fact = 0.5    # decay factor in case of Gradient Descent with Momentum
tmax = 100000       # Max number of iterations
close_initial_guess = False # If true, instead of a random initial guess it
                            # will be close to the solution to make sure the 
                            # model does not get stuck in a local minumum
"""
###############################################################################
###############################################################################
"""




#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
# model_type="logistic"; NFIT=4; xcol=1; ycol=2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2; 
model_type="logistic";   NFIT=4; xcol=2; ycol=0;

#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#EXTRACT AGE<18
if(model_type=="linear"):
	y=y[x[:]<18]; x=x[x[:]<18]; 

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]
xtbatch = xt
ytbatch=yt


#MODEL
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#LOSS FUNCTION
def loss(p):
	global iterations,loss_train,loss_val,iteration, xtbatch, ytbatch

	#TRAINING LOSS
	yp=model(xtbatch,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-ytbatch)**2.0))  #MSE

	#VALIDATION LOSS
	yp=model(xv,p) #model predictions for given parameterization p
	validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	#if(iteration==0):    print("iteration	training_loss	validation_loss") 
	#if(iteration%25==0): print(iteration,"	",training_loss,"	",validation_loss) 
	
	#RECORD FOR PLOTING
	loss_train.append(training_loss); loss_val.append(validation_loss)
	iterations.append(iteration); iteration+=1

	return training_loss

#INITIAL GUESS
po=np.random.uniform(0.1,1.,size=NFIT)

NDIM = len(po)


def optimizer(f, algo="GD", LR=0.0001, method="batch", n_batches=2, decay_fact=0.5, tmax=100000):
    global xtbatch, ytbatch, xt, yt, xi, close_initial_guess
    print("#--------GRADIENT DECENT--------")
    
    #PARAM
    dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
    LR=LR								#LEARNING RATE
    t=0 	 							#INITIAL ITERATION COUNTER
    tmax=tmax							#MAX NUMBER OF ITERATION
    tol=10**-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
    xi = np.random.uniform(0.1,1.,size=NFIT)    # INITIAL GUESS
    if(close_initial_guess): xi = [-3,3,-1,0]   # START WITH A GUESS CLOSE TO THE SOLUTION
    
    # Create a vector of random indexes with the size of the training set
    # to use as auxiliar fot the mini batch method
    if(method=="mini-batch"):
        rand_index_batch = np.random.permutation(xt.shape[0])
    
    print("INITAL GUESS: ",xi)
    
    while(t<=tmax):
    	t=t+1
        
    	if(method == "batch" and t==1):
        	xtbatch = xt; ytbatch = yt
    	if(method == "stochastic"):
        	if(t==1): 
        		index_to_use = 0
        	else:
        		if(index_to_use==len(xt)-1):
        			index_to_use=0
        		else:
        			index_to_use = index_to_use + 1
        	xtbatch=xt[index_to_use]; ytbatch = yt[index_to_use]
    	if(method=="mini-batch"):
            index_from=int(np.floor(len(xt) / n_batches)*((t-1)%n_batches))
            if(t % n_batches == 0):
                index_to = int(len(xt))
            else:
                index_to= int(index_from + np.floor(len(xt) / n_batches))
            index_to_use = rand_index_batch[index_from:index_to]
            xtbatch=xt[index_to_use]; ytbatch = yt[index_to_use]
            
        
    
        
       
       	#NUMERICALLY COMPUTE GRADIENT 
    	df_dx=np.zeros(NDIM)
    	df_dx_prev=0
    	for i in range(0,NDIM):
       		dX=np.zeros(NDIM);
       		dX[i]=dx; 
       		xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
       		df_dx[i]=(f(xi)-f(xm1))/dx
               #print(xi.shape,df_dx.shape)
           
    	if (algo=="GD"):
       	    xip1=xi-LR*df_dx #STEP 
    	elif (algo=="GDM"):
       	    xip1=xi-LR*df_dx - decay_fact*df_dx_prev
       	    df_dx_prev = df_dx
               
               
    	
    	
    	if(method=="batch" and t%10==0):
       	    print(t,"	",xi,"	","	",f(xi)) #,df) 
    	if(method=="stochastic"):
       	    print(t,"	", index_to_use,"	",xi,"	","	",f(xi)) #,df) 
    	if(method=="mini-batch"):
       	    print(t,"	",t%n_batches +1,"	",xi,"	","	",f(xi)) #,df) 
  
        
    	if(t%10==0):
       	    df=np.mean(np.absolute(f(xip1)-f(xi)))
       	    if(df<tol):
       	        print("STOPPING CRITERION MET (STOPPING TRAINING)")
       	        break
                   
                   
       	#UPDATE FOR NEXT ITERATION OF LOOP
    	xi=xip1
   
    return xi

##############################################################################
########################     Optimization Call     ###########################
##############################################################################

#TRAIN MODEL USING SCIPY MINIMIZ 
res = minimize(loss, po, method=OPT_ALGO, tol=1e-15);  popt1=res.x
print("OPTIMAL PARAM:",popt1)

popt = optimizer(loss, algo=algo, LR=LR, method=method, n_batches=n_batches, decay_fact=decay_fact, tmax=tmax)
print("OPTIMAL PARAM:",popt)
print(popt1)





#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()