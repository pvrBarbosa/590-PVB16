# -*- coding: utf-8 -*-
"""
@author: pvict
"""
import os, shutil

###############################################################
################## PARAMETERS #################################
###############################################################

original_dataset_dir = 'G:/DATA/dog_cat/dog_cat_train'
base_dir = 'G:/DATA/dog_cat/dog_cat_small'
create_dir = False          # if True will create Train, Test 
                            # and Validation folders inside base_dir
sample_data = False         # If True will sample the original data from original_dataset_dir
print_data_check = False    # If true will print the amount of samples considered

plot_layers = True          # If true will use the sample image to show the output
                            # of each layer
# Path of the sample image to visually see each layer activation
img_path = 'G:/DATA/dog_cat/dog_cat_small/test/dogs/dog.1710.jpg'


################################################################
############### FOLDER STRUCTURE GENERATION ####################
################################################################
### CREATE THE FOLDER TREE TO RECEIVE A SMALL SUBSET OF THE DATASET
### SPLIT INTO TRAIN, TEST AND VALIDATION

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

def create_directories():
    os.mkdir(base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)
    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)
    os.mkdir(test_cats_dir)
    os.mkdir(test_dogs_dir)


###############################################################


################################################################
##################### DATA SMALL SAMPLING ######################
################################################################
### SAMPLE THE ORIGINAL DATASET TO GET 2000 TRAINING IMAGES,
### 1000 TEST IMAGES AND 1000 VALIDATION. 
### ALL SETS ARE BALANCED, 50% DOG AND 50% CAT 
def create_small_sample():
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
    


def data_check():
    print('total training cat images:', len(os.listdir(train_cats_dir)))
    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('total test cat images:', len(os.listdir(test_cats_dir)))
    print('total test dog images:', len(os.listdir(test_dogs_dir)))
    
    



######################################################################
################## IMPORTING PRE-TRAINED MODEL #######################
######################################################################
# IMPORTING MODEL VGG16 (TRAINED ON IMAGENET DATASET) TO EXTRACT FEATURES
# FROM OUR CAT_DOG IMAGE DATA
from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


# SETTINS GINE TUNING LAYERS
# WE WILL FINE TUNE JUST THE LAST THREE LAYERS OF VGG16
# THE PREVIOUS LAYERS ARE MORE GENERALISTS AND LESS PRONE TO OVERFIT
# THE CODE LOOKS FOR THE LAYER block5_conv1 AND UNFREEZES IT AND EVERYTHING AFTER
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    



##################################################################
######################### MODEL CREATION #########################
##################################################################

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers



model = models.Sequential()
model.add(conv_base) # USING THE VGG16 MODEL
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

model.summary()





##########################################################################
################# READ AND PREPARE/AUGUMENT THE IAMGES ###################
##########################################################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# THIS IMAGE GENERATOR WILL CREATE SEVERAL NEW IMAGES BY SHIFTING, ROTATING, 
# ZOOMING, STRETCHING THE ORIGINAL IMAGES. THIS ENHANCE THE MODEL ACCURACY 
# BECOUSE NOW THE MODEL CAN UNDERSTAND THAT THESE TRANSFORMATIONS DOESNT CHANGE
# THE IMAGE CLASSIFICATION
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])




##########################################################################
####################### FIT AND SAVE THE MODEL ###########################
##########################################################################

history = model.fit_generator(train_generator,
                              steps_per_epoch=int(2000/20), # NUM OF IMAGES / BATCH SIZE
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=int(1000/20))



#model.save('cats_and_dogs_small.h5')


##########################################################################
################# PLOT MODEL ACCURACY AND LOSS ###########################
##########################################################################

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
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



##########################################################################
############# PLOT SMOOTHED MODEL ACCURACY AND LOSS ######################
##########################################################################

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



##########################################################################
############# TEST THE MODEL AND SHOW THE ACCURACY #######################
##########################################################################

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)





##########################################################################
############# DISPLAY THE IMAGE ACTIVATION IN EACH LAYER #################
##########################################################################

def plot_layer_output():
    # IMPORT ONE IMAGE TO ANALYZE THE FEATURES
    img_path = 'G:/DATA/dog_cat/dog_cat_small/test/dogs/dog.1710.jpg'
    
    from keras.preprocessing import image
    import numpy as np
    from keras import models
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    
    # CREATE A SIMPLER MODEL WITHOUT IMAGE AUGUMENTATION, BUT WITH DROPOUT
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
    input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'])
    
    
    
    # CREATE A VARIABLE THAT STORES THE OUTPUT OF EACH LAYER
    layer_outputs = [layer.output for layer in model.layers[:8]]
    # CREATE A NEW MODEL THAT CAN OUTPUT THE LAYER OUTPUTS OF THE ORIGINAL MODEL 
    # FOR A GIVEN INPUT
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    
    
    # SHOW THE ORIGINAL IMAGE
    import matplotlib.pyplot as plt
    plt.imshow(img_tensor[0])
    plt.show()
    
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

#############################################################################

if (create_dir): create_directories()                               
if (sample_data): create_small_sample()        
if (print_data_check): data_check() 
if (plot_layers): plot_layer_output()          















