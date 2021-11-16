import numpy as np
import pandas as pd
import seaborn as sns

import keras
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from tensorflow.keras.losses import msle
from tensorflow.keras.losses import mse
from sklearn.metrics import accuracy_score
from random import randrange
from random import sample


#GET DATASET
from keras.datasets import mnist
from keras.datasets import fashion_mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#-----------------------------------------------------------------------------
######################## PARAMETERS #########################################
#-----------------------------------------------------------------------------
epochs = 50
batch_size = 1000
dff_layers = [512, 256, 128, 32, 128, 256, 512]
activation_func = "relu"

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
#NORMALIZE AND RESHAPE
#-----------------------------------------------------------------------------
img_size = len(X[0])
X=X/np.max(X) 
X=X.reshape(len(X),img_size*img_size)

test_images=test_images/np.max(test_images) 
test_images=test_images.reshape(len(test_images),img_size*img_size)

#-----------------------------------------------------------------------------
#MODEL
#-----------------------------------------------------------------------------

model = models.Sequential()
model.add(layers.Dense(dff_layers[0], activation=activation_func, input_shape=(img_size * img_size,)))
for l in dff_layers[1:]:
        model.add(layers.Dense(l, activation=activation_func))
model.add(layers.Dense(img_size * img_size,  activation='linear'))


#-----------------------------------------------------------------------------
#COMPILE AND FIT
#-----------------------------------------------------------------------------
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()
history = model.fit(X, X, epochs=epochs, batch_size=batch_size,validation_split=0.2)


#-------------------------------------------------------------------------
# PLOT TRAN/VAL LOSS
#-------------------------------------------------------------------------
# Code form HW4.0
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#-----------------------------------------------------------------------------
#EVALUATE ON TEST DATA
#-----------------------------------------------------------------------------

print("----------------- Model MSE --------------------")
print("Train MSE:")
model.evaluate(X, X)
print("Test MSE:")
model.evaluate(test_images,test_images)


#-----------------------------------------------------------------------------
#PLOT ORIGINAL AND RECONSTRUCTED 
#-----------------------------------------------------------------------------
def plot_orig_rec(model, X, title = ""):
    X1=model.predict(X) 
    
    #RESHAPE
    size = int(np.sqrt(len(X[0])))
    length = len(X)
    X=X.reshape(length,size,size)
    X1=X1.reshape(length,size,size)
    
    #COMPARE ORIGINAL 
    f, ax = plt.subplots(4,1)
    I1=randrange(len(X)); I2=randrange(len(X))
    f.suptitle(title)
    ax[0].imshow(X[I1])
    ax[1].imshow(X1[I1])
    ax[2].imshow(X[I2])
    ax[3].imshow(X1[I2])
    plt.show()


# Load FASHION_MNIST to test image reconstruction
(X_f, Y_f), (test_images_f, test_labels_f) = fashion_mnist.load_data()
X_f=X_f/np.max(X_f) 
X_f=X_f.reshape(len(X_f),img_size*img_size)

plot_orig_rec(model, X, "MNIST Reconstruction")
plot_orig_rec(model, X_f, "FASHION MNIST Reconstruction")

#-------------------------------------------------------------------------
# ANOMALY DETECTION 
#-------------------------------------------------------------------------
# Code adaptation from https://www.analyticsvidhya.com/blog/2021/05/anomaly-detection-using-autoencoders-a-walk-through-in-python/
def detect_anomaly(model, test_data, trained_data):

    train_pred = model.predict(trained_data)
    # get the error for each instance
    train_errors = msle(train_pred, trained_data)
    # Calculates the threshold as mean + 4stddev
    threshold = np.mean(train_errors.numpy()) + 4*np.std(train_errors.numpy())
        
    test_pred = model.predict(test_data)
    # get the error for each instance
    test_errors = msle(test_pred, test_data)
    # 1 = anomaly, 0 = normal
    anomaly_mask = pd.Series(test_errors) > threshold
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0)
    return preds
    

# Create a dataset mixed between MNIST and FASION_MNIST to check for anomaly
# get 800 images from original data
original_sample = X[sample(range(len(X)), 1000)] 
# get 200 anomaly images
anomaly_sample = X_f[sample(range(len(X_f)), 1000)] 
# Append the two samples
check_images = np.append(original_sample, anomaly_sample, axis = 0)

# Create a list where 0 is an original image and 1 is an anomaly
original_label = [0]*len(original_sample)
anomaly_label = [1]*len(anomaly_sample)
check_label = np.append(original_label, anomaly_label, axis = 0)

# Call the anomaly detection function
predictions = detect_anomaly(model, check_images, X)

train_detect = detect_anomaly(model, X, X)
train_acc = 1-(sum(train_detect)/len(train_detect))

anom_detect = detect_anomaly(model, X_f, X)
anom_acc = sum(anom_detect)/len(anom_detect)


# Calculate the accuracy
print("--------- ANOMALY DETECTION ACCURACY -------------")
print("Acc on train dataset:" + str(train_acc))
print("Acc on anomaly dataset:" + str(anom_acc))
print("Acc on mixed dataset:" + str(accuracy_score(predictions, check_label)))


#-------------------------------------------------------------------------
# SAVE MODEL
#-------------------------------------------------------------------------
model.save("HW6.1-model.h5")

