# after pre-train veries model with 65% data
# 35% data remain will help us to evaluate weighting ratio for ensembling 

import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import optimizers, backend
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, Activation
from keras.models import load_model
from keras.layers import RepeatVector, TimeDistributed, LSTM, GRU, Flatten, Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.utils.generic_utils import get_custom_objects

from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, adam, Nadam

train = pd.read_csv("35%_data.csv")
train_data = train[ features ]

real_updown = [None]
for i in range(len(train_data)-1):
    if train_data.iloc[i+1,0]-train_data.iloc[i,0] >= 0:
        real_updown.append(1)
    else:
        real_updown.append(0)
        
def swish(x):
    return ((K.sigmoid(x) * x * 1.5))
get_custom_objects().update({'swish': Activation(swish)})
        

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)

#scaler = StandardScaler()
#train_data = scaler.fit_transform(train_data)


window = 5
forecast_day = 7
training_data_rate = 0.95
feature = train_data.shape[1]

train_x = np.empty(shape=(0, window, feature))
train_y = np.empty(shape=(0, forecast_day))
for i in range(len(train_data)-window-forecast_day+1):
    train_x = np.vstack((train_x, train_data[np.newaxis,i:(i+window),:]))
    train_y = np.vstack((train_y, train_data[(i+window):(i+window+forecast_day), 0]))


######## creat training dataset from predictions from 5 different deep learning model  ##########

# validation_y to array
val_y = np.zeros(shape=(len(train_y), forecast_day))
for i in range(len(train_y)):
    val_y[i,:] = train_y[i,:]

# 1st model
model = load_model("35%_relu_1.h5")
train_set_pred = model.predict(train_x, batch_size=4)
predictions = np.zeros(shape=(len(train_y), 5, forecast_day))

for i in range(len(train_y)):
    #value of prediction
    predictions[i,0,:] = train_set_pred[i,:]

# 2nd model
model = load_model("35%_swish_2.h5")
train_set_pred = model.predict(train_x, batch_size=4)

for i in range(len(train_y)):
    #value of prediction
    predictions[i,1,:] = train_set_pred[i,:]

# 3rd model
model = load_model("35%_selu_1.h5")
train_set_pred = model.predict(train_x, batch_size=4)

for i in range(len(train_y)):
    #value of prediction
    predictions[i,2,:] = train_set_pred[i,:]

# 4th model
model = load_model("35%_lstm_1.h5")
train_set_pred = model.predict(train_x, batch_size=4)

for i in range(len(train_y)):
    #value of prediction
    predictions[i,3,:] = train_set_pred[i,:]

# 5th model
model = load_model("35%_lstm_2.h5")
train_set_pred = model.predict(train_x, batch_size=4)

for i in range(len(train_y)):
    #value of prediction
    predictions[i,4,:] = train_set_pred[i,:]
print(predictions[1])

#######################################################

train_x = predictions
train_y = val_y

train_x, validation_x = train_x[:int(len(train_x)*training_data_rate),:,:], train_x[int(len(train_x)*training_data_rate):,:,:] 
train_y, validation_y = train_y[:int(len(train_y)*training_data_rate),:], train_y[int(len(train_y)*training_data_rate):,:]

# main, train stacking model with not too complex model

model = Sequential()
model.add(Conv1D(nb_filter=64, filter_length=1, input_shape=(5, 7), activation="linear"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(forecast_day))

adam = optimizers.Adam(lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=1e-6,
    amsgrad=False)

nadam = optimizers.Nadam(lr=0.01, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=None, 
    schedule_decay=0.001)

model.compile(
    loss = "mean_squared_logarithmic_error",
    optimizer = adam)
    #metrics = ["mean_squared_logarithmic_error"])

checkpoint = ModelCheckpoint(
    filepath = "stacking.h5",
    monitor = "val_loss",
    verbose = 1,
    save_best_only = True,
    mode = "min")

earlystopping = EarlyStopping(
    monitor = "val_loss",
    patience = 30,
    verbose = 1,
    mode = "auto")

start = timeit.default_timer()

train_history = model.fit(
    x=train_x,
    y=train_y,
    epochs=2500,
    validation_data=(validation_x, validation_y),
    batch_size=8,
    shuffle=False,
    verbose=2,
    callbacks = [checkpoint, earlystopping])


# load best saved weight
model = load_model("stacking_.h5")
validation_set_pred = model.predict(validation_x, batch_size=4)

validation_set_pred = validation_set_pred * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]
validation_y = validation_y * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]

predictions = np.zeros(shape=(len(validation_y), forecast_day, 5))
for i in range(len(validation_y)):
    #value of prediction
    predictions[i,:,0] = validation_set_pred[i,:]
    #actual value
    predictions[i,:,1] = validation_y[i,:].reshape(1,forecast_day)
    #prediction of trend
    predictions[i,1:,2] = (predictions[i,1:,0] - predictions[i,:-1,1]) >= 0
    #actual trend
    predictions[i,:,3] = real_updown[int(len(train_x))+window:][i:i+forecast_day]
for i in range(1,len(predictions)):
    predictions[i,0,2] = (predictions[i,0,0] - predictions[i-1,0,1] > 0)
predictions[0,0,2] = (predictions[0,0,0] - train.iloc[int(len(train_x))+window-1, 1]) > 0
for i in range(len(predictions)):
    #weather prediction of trend is accurate or not
    predictions[i,:,4] = (predictions[i,:,2] == predictions[i,:,3])
    
for i in range(forecast_day):
    print("Trend accuracy of", i+1,"days after：", predictions[:,i,4].mean())
print("7d average trend accuracy：", sum([predictions[i,:,4].mean() for i in range(len(predictions))])/len(predictions))

# saved stacking weight then retrain model with 100% data
