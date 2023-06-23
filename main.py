import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from train import LinearRegression

# features = np.array([1,2,3,5,6,7])
# labels = np.array([155, 197, 244, 356,407,448])

# model = LinearRegression()






from train_multiple import MultipleLinearRegression

df =  pd.read_csv('advertsing.csv') ## Rename with your's csv file name

# Perform feature scaling
features = df[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
target = df['Sales ($)']

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features) ## While performing feature scaling, the learning rate 0.01 is enough to get better results
# without feature scale, the learning rate 0.0000001 should be used to get better results.

features = pd.DataFrame(scaled_features, columns=features.columns)
features = np.array(features)
labels = np.array(target)

model = MultipleLinearRegression()

model.train(x_train = features, y_train = labels, epochs=250)

