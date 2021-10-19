
#CODE MODIFIED FROM:
# chollet-deep-learning-in-python (p27-30)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame

##------------------------
##GET AND BRIEFLY EXPLORE IMAGE
##------------------------

####GET DATASET
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()




#QUICK INFO ON IMAGE
def get_info(image):
	print("\n------------------------")
	print("INFO")
	print("------------------------")
	print("SHAPE:",image.shape)
	print("MIN:",image.min())
	print("MAX:",image.max())
	print("TYPE:",type(image))
	print("DTYPE:",image.dtype)
#	print(DataFrame(image))

get_info(train_images)
get_info(train_labels)

##------------------------
#SET UP MODEL AND DATA 
##------------------------


from keras import models
from keras import layers


#INITIALIZE MODEL	
	# Sequential model --> plain stack of layers 	
	# each layer has exactly one input tensor and one output tensor.
network = models.Sequential()

#ADD LAYERS
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

#SOFTMAX  --> 10 probability scores (summing to 1
network.add(layers.Dense(10,  activation='softmax'))

#COMPILATION (i.e. choose optimizer, loss, and metrics to monitor)
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#PREPROCESS THE DATA

#UNWRAP 28x28x MATRICES INTO LONG VECTORS (784,1) #STACK AS BATCH
train_images = train_images.reshape((NKEEP, 28 * 28)) 
#RESCALE INTS [0 to 255] MATRIX INTO RANGE FLOATS RANGE [0 TO 1] 
#train_images.max()=255 for grayscale
train_images = train_images.astype('float32') / train_images.max() 

#REPEAT FOR TEST DATA
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / test_images.max()

#DEBUGGING
NKEEP=60000
batch_size=int(0.1*NKEEP)
train_images=train_images[0:NKEEP,:,:]
train_labels=train_labels[0:NKEEP]

from keras.utils import to_categorical

#keras.utils.to_categorical does the following
# 5 --> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# 1 --> [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 9 --> [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.] ... etc
#print(train_labels[2])

train_labels = to_categorical(train_labels); #print(train_labels[2])
test_labels = to_categorical(test_labels)

##------------------------
#TRAIN AND EVALUTE
##------------------------

#TRAIN
epochs=30 
network.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

# #EVALUTE
train_loss, train_acc = network.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = network.evaluate(test_images, test_labels,batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)






