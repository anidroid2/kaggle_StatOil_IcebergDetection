#test data rgb
#submission data

import numpy as np 				# linear algebra
import pandas as pd 			# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle
from scipy.ndimage import rotate

df_test = pd.read_json('Data/test.json')

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
rgb_test = color_composite(df_test)
data_dict = {'id': df_test['id'], 'rgb_test': rgb_test}

with open("Data/data_test", 'wb') as f:
	pickle.dump(data_dict, f)
