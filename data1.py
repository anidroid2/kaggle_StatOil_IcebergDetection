#original + flipped data +rgb

import numpy as np 				# linear algebra
import pandas as pd 			# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle
from scipy.ndimage import rotate

df_train = pd.read_json('Data/train.json')

def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)


#convert to RGB images
rgb_train = color_composite(df_train)
#Rotate images to create more data samples
a = rotate(rgb_train, angle =90,axes = (1,2))
b = rotate(rgb_train, angle =180,axes = (1,2))
c = rotate(rgb_train, angle =270,axes = (1,2))
#Data augmentation
X = np.concatenate([rgb_train,a,b,c],axis = 0)
Y_s = df_train['is_iceberg'].values
Y = np.concatenate([Y_s,Y_s,Y_s,Y_s])
Y =np.matrix(Y)
Y=Y.transpose()
#shuffle the data
X, Y = shuffle(X, Y, random_state=0)
data_dict = {'X' : X, 'Y' : Y}

with open("Data/data1", 'wb') as f:
	pickle.dump(data_dict, f)