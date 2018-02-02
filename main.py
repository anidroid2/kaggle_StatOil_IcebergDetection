#kernel reference: https://www.kaggle.com/cbryant/keras-cnn-statoil-iceberg-lb-0-1995-now-0-1516

import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 

# Import Keras 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

train = 100   #INPUT HERE     #Number of epoch's to train

model_number ="1"
model_path = "Models/Model" + model_number + '/'
h5_filepath = model_path
pickle_filepath = model_path + 'picklefile'

metric = {'acc':[],'val_acc':[],'loss':[],'val_loss':[]}
if (os.path.isfile(pickle_filepath)):
	with open(pickle_filepath,'rb') as f:
		metric = pickle.load(f)



#get training data
with open('Data/data1','rb') as f:   #original size #bands data
	data_dict1 = pickle.load(f)

X = data_dict1['X']
Y = data_dict1['Y']

del data_dict1


print(np.shape(X))
print(np.shape(Y))

#Build keras model    
model=Sequential()
    
    # CNN 1
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.2))

    # CNN 2
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

    # CNN 3
model.add(Conv2D(128, kernel_size=(3, 3),padding = "same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

    #CNN 4
model.add(Conv2D(128, kernel_size=(3, 3),padding = "same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Dropout(0.3))

    # You must flatten the data for the dense layers
model.add(Flatten())

    #Dense 1
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

    #Dense 2
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

    # Output 
model.add(Dense(1, activation="sigmoid"))

optimizer = Adam(lr=0.0001, decay =1e-6 )
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

if(train ==0):
  epoch_sel = 30
  model.load_weights(h5_filepath + str(epoch_sel), by_name=False)

#model.load_weights(h5_filepath + str(28), by_name=False)

class AccuracyHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        metric['acc'].append(logs.get('acc'))
        metric['loss'].append(logs.get('loss'))
        metric['val_acc'].append(logs.get('val_acc'))
        metric['val_loss'].append(logs.get('val_loss'))
        if((batch > 0) and (batch % 2 == 0) ):
        	model.save_weights(h5_filepath  +str(batch))

history = AccuracyHistory()

x_train = X[0:4000,:,:,:]
x_test = X[4000:,:,:,:]

#y_cat = (to_categorical((Y)))
y_train = (Y)[0:4000,:]
y_test = (Y)[4000:,:]

del X

if train:
	model.fit(x_train, y_train,
          batch_size=100,
          epochs=train,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history]
          )

#model.save_weights(h5_filepath)
with open(pickle_filepath, 'wb') as f:
	pickle.dump(metric, f)

plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot(metric['acc'])
plt.plot(metric['val_acc'])
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy")

plt.subplot(212)             # the first subplot in the first figure
plt.plot(metric['loss'])
plt.plot(metric['val_loss'])
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss")
plt.show()

if(train ==0): 
	#Working on test set
	with open('Data/data_test','rb') as f:
		data_dict2 = pickle.load(f)

	predictions = model.predict(data_dict2['rgb_test'])
	rounded = [(x[0]) for x in predictions]
	op_dict =  {'id': data_dict2['id'], 'is_iceberg': rounded }
	pd_op = pd.DataFrame(op_dict)
	pd_op.to_csv('op.csv',index = False)