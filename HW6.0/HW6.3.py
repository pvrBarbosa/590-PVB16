

#SOURCE: MODIFIED FROM https://blog.keras.io/building-autoencoders-in-keras.html

import keras
from keras import layers
import matplotlib.pyplot as plt
import ssl
from keras.datasets import mnist,cifar10, fashion_mnist, cifar100
import numpy as np
import pandas as pd
from random import randrange
from random import sample
from tensorflow.keras.losses import msle
from tensorflow.keras.losses import mse
from sklearn.metrics import accuracy_score
ssl._create_default_https_context = ssl._create_unverified_context


#-----------------------------------------------------------------------------
#USER PARAM
#-----------------------------------------------------------------------------
EPOCHS          =   35
NKEEP           =   50000        #MNIST total = 60000 / CIFAR10 total = 50000
BATCH_SIZE      =   128
DATA            =   "CIFAR"


#-----------------------------------------------------------------------------
#GET DATA
#-----------------------------------------------------------------------------
if(DATA=="MNIST"):
    (x_train, _), (x_test, _) = mnist.load_data()
    N_channels=1; PIX=28

if(DATA=="CIFAR"):
    (x_train, _), (x_test, _) = cifar10.load_data()
    N_channels=3; PIX=32
    EPOCHS=100 #OVERWRITE

#-----------------------------------------------------------------------------
#NORMALIZE AND RESHAPE
#-----------------------------------------------------------------------------
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#-----------------------------------------------------------------------------
#DOWNSIZE TO RUN FASTER AND DEBUG
#-----------------------------------------------------------------------------
print("BEFORE",x_train.shape)
x_train=x_train[0:NKEEP]
x_test=x_test[0:NKEEP]
print("AFTER",x_train.shape)

#-----------------------------------------------------------------------------
#BUILD CNN-AE MODEL
#-----------------------------------------------------------------------------
if(DATA=="MNIST"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    # #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # # AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
 
    # #DECODER
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)


if(DATA=="CIFAR"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    #DECODER
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)



#-----------------------------------------------------------------------------
#COMPILE
#-----------------------------------------------------------------------------
model = keras.Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy');
model.summary()

#-----------------------------------------------------------------------------
#TRAIN
#-----------------------------------------------------------------------------
history = model.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                )


#-----------------------------------------------------------------------------
#HISTORY PLOT
#-----------------------------------------------------------------------------
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()


#-----------------------------------------------------------------------------
#EVALUATE ON TEST DATA
#-----------------------------------------------------------------------------

print("----------------- Model MSE --------------------")
print("Train MSE:")
model.evaluate(x_train, x_train)
print("Test MSE:")
model.evaluate(x_test,x_test)


#-----------------------------------------------------------------------------
#MAKE PREDICTIONS FOR TEST DATA
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
#PLOT ORIGINAL AND RECONSTRUCTED 
#-----------------------------------------------------------------------------
def plot_orig_rec(model, X, title = ""):
    
    decoded_imgs = model.predict(X)
    n = 5
    plt.figure(figsize=(5, 2))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(X[i].reshape(PIX, PIX,N_channels))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(PIX, PIX,N_channels))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.suptitle(title)
    plt.show()


# Load CIFAR100 to test image reconstruction
(X_c, Y_c), (test_images_100, test_labels_100) = cifar100.load_data()

# Remove class "truck" from CIFAR100 (truck has the label == 58)
X_c = X_c[Y_c.reshape(len(Y_c))!=58, :, :, :]

plot_orig_rec(model, x_train, "CIFAR10 Reconstruction")
plot_orig_rec(model, X_c, "CIFAR100 Reconstruction")

#-------------------------------------------------------------------------
# ANOMALY DETECTION 
#-------------------------------------------------------------------------
# Code adaptation from https://www.analyticsvidhya.com/blog/2021/05/anomaly-detection-using-autoencoders-a-walk-through-in-python/
def detect_anomaly(model, test_data, trained_data):

    train_pred = model.predict(trained_data)
    # get the error for each instance
    train_errors = np.mean(np.mean(msle(train_pred.reshape(trained_data.shape), trained_data).numpy(),axis =1),axis =1)
    # Calculates the threshold as mean + 4stddev
    threshold = np.mean(train_errors + 4*np.std(train_errors))
        
    test_pred = model.predict(test_data)
    # get the error for each instance
    test_errors = np.mean(np.mean(msle(test_pred.reshape(test_data.shape), test_data).numpy(),axis =1), axis=1)
    # 1 = anomaly, 0 = normal
    anomaly_mask = pd.Series(test_errors) > threshold
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0)
    return preds
    

#test_data = check_images
#trained_data = x_train[:5000]


# Create a dataset mixed between MNIST and FASION_MNIST to check for anomaly
# get 800 images from original data
original_sample = x_train[sample(range(len(x_train)), 1000)] 
# get 200 anomaly images
anomaly_sample = X_c[sample(range(len(X_c)), 1000)] 
# Append the two samples
check_images = np.append(original_sample, anomaly_sample, axis = 0)

# Create a list where 0 is an original image and 1 is an anomaly
original_label = [0]*len(original_sample)
anomaly_label = [1]*len(anomaly_sample)
check_label = np.append(original_label, anomaly_label, axis = 0)

# Call the anomaly detection function
# (Had to reduce to 5000 data points for my memory to handle this)
predictions = detect_anomaly(model, check_images, x_train[:5000])

train_detect = detect_anomaly(model, x_train[:5000], x_train[:5000])
train_acc = 1-(sum(train_detect)/len(train_detect))


# Calculate the accuracy
print("--------- ANOMALY DETECTION ACCURACY -------------")
print("Acc on train dataset:" + str(train_acc))
print("Acc on mixed dataset:" + str(accuracy_score(predictions, check_label)))


#-------------------------------------------------------------------------
# SAVE MODEL
#-------------------------------------------------------------------------
model.save("HW6.3-model.h5")







