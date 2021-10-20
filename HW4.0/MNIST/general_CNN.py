# -*- coding: utf-8 -*-
"""
@author: pvict
"""

#-----------------------------------------------------------------------------
##############################    PARAMETERS     #############################
#-----------------------------------------------------------------------------

dataset = "FASHION"         # MNIST / FASHION / CIFAR10
model_type = "CNN"          # CNN / DFF
epochs = 50                 # number of epochs
data_augument = True        

####################### CNN MODEL PARAMETERS #######################
# Configure the parameters of the first convolutional layer + maxpool
cnn_1L_filter = 32          # Number of filters
cnn_1L_kernel = 3           # kernel size
cnn_1L_maxpool = 2          # maxpool size
cnn_1L_activation = "relu"  # activation function

# Configure the parameters of the n intermediate layers + maxpool
cnn_nIL = 1                 # number of intermediate convolutional layers + maxpool
cnn_IL_filter = 64          # number of filters
cnn_IL_kernel = 2           # kernel size
cnn_IL_maxpool = 2          # maxpool size
cnn_IL_activation = "relu"  # activation function

# Configure the amount of filters in the last layer
cnn_dropout = 0.3           # dropout percentage
cnn_LL_neurons = 64         # number of nodes in the DNN
cnn_LL_activation = "relu"  # activation function


####################### DFF MODEL PARAMETERS #######################
dff_layers = [512, 256]     # array where each position represents the amount
                            # of neurons in the corresponding layer
dff_activation = "relu"     # activation function

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------




from keras import layers 
from keras import models
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------

if (dataset == "CIFAR10"): 
    img_size = 32
else:
    img_size = 28

# Create the CNN model based on the selected parameters
if (model_type == "CNN"):
    model = models.Sequential()
    model.add(layers.Conv2D(cnn_1L_filter, (cnn_1L_kernel, cnn_1L_kernel), 
                            activation = cnn_1L_activation, 
                            input_shape = (img_size, img_size, 1)))
    model.add(layers.MaxPooling2D((cnn_1L_maxpool, cnn_1L_maxpool)))
    
    for i in range(cnn_nIL):
        model.add(layers.Conv2D(cnn_IL_filter, (cnn_IL_kernel, cnn_IL_kernel), 
                                activation=cnn_IL_activation)) 
        model.add(layers.MaxPooling2D((cnn_IL_maxpool, cnn_IL_maxpool)))

    
    model.add(layers.Flatten())
    model.add(layers.Dropout(cnn_dropout))
    model.add(layers.Dense(cnn_LL_neurons, activation=cnn_LL_activation))
    model.add(layers.Dense(10, activation='softmax'))
    
# Create the DFF model based on the selected parameters 
if (model_type == "DFF"):
    model = models.Sequential()
    for l in dff_layers:
        model.add(layers.Dense(l, activation=dff_activation, 
                               input_shape=(img_size * img_size,)))
    
    model.add(layers.Dense(10,  activation='softmax'))


model.summary()
    


#-------------------------------------
#GET DATA AND REFORMAT
#-------------------------------------
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


if dataset == "MNIST": data = mnist
elif dataset == "FASHION": data = fashion_mnist
elif dataset == "CIFAR10": data = cifar10

# Load the selected dataset
(train_images, train_labels), (test_images, test_labels) = data.load_data()


# Split training data into train / validation (80/20)
train_images, validation_images, \
    train_labels, validation_labels = train_test_split(train_images, 
                                                       train_labels, 
                                                       test_size=0.20)



#----------------------------------------
#NORMALIZE AND AUGUMENT DATA
#---------------------------------------
NKEEP=len(train_images)
batch_size=int(0.05*NKEEP)

if model_type == "CNN":
    
    # Reshape the images
    train_images = train_images.reshape((list(train_images.shape) + [1]))
    test_images = test_images.reshape((list(test_images.shape) + [1]))
    validation_images = validation_images.reshape((list(validation_images.shape) + [1]))
    
    
    # Create the augumentation function if this parameter is true
    if (data_augument == True):
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Rescale image generator
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_datagen.fit(train_images)
    validation_datagen.fit(validation_images)
    test_datagen.fit(test_images)
    
    
    #CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    validation_labels = to_categorical(validation_labels)
    
    
    train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
    validation_generator = validation_datagen.flow(validation_images, validation_labels, batch_size=batch_size)
    test_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)



if model_type == "DFF":
    #UNWRAP 28x28x MATRICES INTO LONG VECTORS (784,1) #STACK AS BATCH
    train_images = train_images.reshape((len(train_images), img_size * img_size)) 
    #RESCALE INTS [0 to 255] MATRIX INTO RANGE FLOATS RANGE [0 TO 1] 
    #train_images.max()=255 for grayscale
    train_images = train_images.astype('float32') / train_images.max() 
    
    #REPEAT FOR TEST DATA
    test_images = test_images.reshape((len(test_images), img_size * img_size))
    test_images = test_images.astype('float32') / test_images.max()
    
    #REPEAT FOR VALIDATION DATA
    validation_images = validation_images.reshape((len(validation_images), img_size * img_size))
    validation_images = validation_images.astype('float32') / validation_images.max()
    

    
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    validation_labels = to_categorical(validation_labels)








#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if model_type == "CNN":
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=int(len(train_images)/batch_size), # NUM OF IMAGES / BATCH SIZE
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=int(len(validation_images)/batch_size))

if model_type == "DFF": 
    history = model.fit(train_images, train_labels, 
                        validation_data = (validation_images,validation_labels),
                        epochs=epochs, batch_size=batch_size)




#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
if model_type == "CNN":
    train_loss, train_acc = model.evaluate_generator(train_generator, steps=50)
    validation_loss, validation_acc = model.evaluate_generator(validation_generator, steps=50)
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)


if model_type == "DFF":
    train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(test_images, test_labels,batch_size=test_images.shape[0])
    validation_loss, validation_acc = model.evaluate(validation_images, validation_labels,batch_size=validation_images.shape[0])

print('train_acc:', train_acc)
print('validation_acc:', validation_acc)
print('test acc:', test_acc)







#-------------------------------------------------------------------------
################# PLOT MODEL ACCURACY AND LOSS ###########################
#-------------------------------------------------------------------------

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



#-------------------------------------------------------------------------
############# PLOT SMOOTHED MODEL ACCURACY AND LOSS ######################
#-------------------------------------------------------------------------

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




#-------------------------------------------------------------------------
############# DISPLAY A RANDOM IMAGE FROM THE DATASET ####################
#-------------------------------------------------------------------------
from random import randrange
import matplotlib.pyplot as plt
def display_image():
    # Select a random image from the training set
    img = train_images[randrange(len(train_images))]
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # SHOW THE ORIGINAL IMAGE
    plt.imshow(img_tensor[0])
    plt.show()

    
def layer_outputs():
    # Only works for CNN
    if model_type == "DFF": return
    
    # Select a random image from the training set
    img = train_images[randrange(len(train_images))]
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # CREATE A VARIABLE THAT STORES THE OUTPUT OF EACH LAYER
    layer_outputs = [layer.output for layer in model.layers[:8]]
    # CREATE A NEW MODEL THAT CAN OUTPUT THE LAYER OUTPUTS OF THE ORIGINAL MODEL 
    # FOR A GIVEN INPUT
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    
    
    # GET THE LAYER NAMES TO USE INTIDE THE PLOTS
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
        
    # DEFINE THE SIZE OF THE GRID OF IMAGES
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1] # layer_activation.shape = (1, size, size, n_features)
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            # THIS FOR LOOP POSTPROCESS THE IMAGES TO MAKE IR VISIBLE
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :,col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


# Function to save current model
def save_model(name = "MNIST_general_CNN.h5"):
    model.save(name)
    
# Function to load a model
from keras.models import load_model
def load_model(path):
    model = load_model(path)
    return model





