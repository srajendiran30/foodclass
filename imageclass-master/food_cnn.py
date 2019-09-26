from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os

#Initialize the CNN for categorical classification
classifier = Sequential()
#Convolution and Max pooling
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
classifier.add(Flatten())

#Full connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(3, activation = 'softmax'))

#Compile classifier
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('./dataset/class/training_set', target_size=(128, 128), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('./dataset/class/test_set', target_size=(128, 128), batch_size=32, class_mode='categorical')
classifier.fit_generator(training_set, steps_per_epoch=800/32, epochs=50, validation_data=test_set, validation_steps = 200/32)

#save model
import os
target_dir = './models/class'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifier.save('./models/class/model.h5')
classifier.save_weights('./models/class/weights.h5')








#Initialize the CNN
classifierc = Sequential()
#Convolution and Max pooling
classifierc.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifierc.add(MaxPooling2D(pool_size = (2,2)))
classifierc.add(Conv2D(64, (3, 3), activation = 'relu'))
classifierc.add(MaxPooling2D(pool_size = (2,2)))
classifierc.add(Conv2D(128, (3, 3), activation = 'relu'))
classifierc.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
classifierc.add(Flatten())

#Full connection
classifierc.add(Dense(128, activation = 'relu'))
classifierc.add(Dense(2, activation = 'softmax'))

#Compile classifier
classifierc.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('./dataset/food/training_set/cooked', target_size=(128, 128), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('./dataset/food/test_set/cooked', target_size=(128, 128), batch_size=32, class_mode='categorical')
classifierc.fit_generator(training_set, steps_per_epoch=800/32, epochs=50, validation_data=test_set, validation_steps = 200/32)

#save model
import os
target_dir = './models/food/cooked'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifierc.save('./models/food/cooked/model.h5')
classifierc.save_weights('./models/food/cooked/weights.h5')






#Initialize the CNN
classifierf = Sequential()
#Convolution and Max pooling
classifierf.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifierf.add(MaxPooling2D(pool_size = (2,2)))
classifierf.add(Conv2D(64, (3, 3), activation = 'relu'))
classifierf.add(MaxPooling2D(pool_size = (2,2)))
classifierf.add(Conv2D(128, (3, 3), activation = 'relu'))
classifierf.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
classifierf.add(Flatten())

#Full connection
classifierf.add(Dense(128, activation = 'relu'))
classifierf.add(Dense(2, activation = 'softmax'))

#Compile classifier
classifierf.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('./dataset/food/training_set/fries', target_size=(128, 128), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('./dataset/food/test_set/fries', target_size=(128, 128), batch_size=32, class_mode='categorical')
classifierf.fit_generator(training_set, steps_per_epoch=800/32, epochs=50, validation_data=test_set, validation_steps = 200/32)

#save model
import os
target_dir = './models/food/fries'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifierf.save('./models/food/fries/model.h5')
classifierf.save_weights('./models/food/fries/weights.h5')
















#Initialize the CNN
classifierfr = Sequential()
#Convolution and Max pooling
classifierfr.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifierfr.add(MaxPooling2D(pool_size = (2,2)))
classifierfr.add(Conv2D(64, (3, 3), activation = 'relu'))
classifierfr.add(MaxPooling2D(pool_size = (2,2)))
classifierfr.add(Conv2D(128, (3, 3), activation = 'relu'))
classifierfr.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
classifierfr.add(Flatten())

#Full connection
classifierfr.add(Dense(128, activation = 'relu'))
classifierfr.add(Dense(2, activation = 'softmax'))

#Compile classifier
classifierfr.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('./dataset/food/training_set/fruits', target_size=(128, 128), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('./dataset/food/test_set/fruits', target_size=(128, 128), batch_size=32, class_mode='categorical')
classifierfr.fit_generator(training_set, steps_per_epoch=800/32, epochs=50, validation_data=test_set, validation_steps = 200/32)

#save model
import os
target_dir = './models/food/fruits'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifierfr.save('./models/food/fruits/model.h5')
classifierfr.save_weights('./models/food/fruits/weights.h5')





