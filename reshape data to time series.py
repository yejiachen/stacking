# reshape daily index data to time series prediction format

# number of training data = samples*training_data_rate = a
# number of validation_y data = samples*(1-training_data_rate) = b

# train_x size: (a, 5, features ), train_y size: (a, 7)
# validation_x size: (b, 5, features ), validation_y size: (b, 7)

import numpy as np

def reshape_data(raw_data, feature, window, forecast_day, training_data_rate, scaler):

    train = pd.read_csv(raw_data)
    train = train[::-1]
    train_data = train[ feature ]

    real_updown = [None]
    for i in range(len(train_data)-1):
        if train_data.iloc[i+1,0]-train_data.iloc[i,0] >= 0:
            real_updown.append(1)
        else:
            real_updown.append(0)

    train_data = scaler.fit_transform(train_data)
    feature_n = train_data.shape[1]

    train_x = np.empty(shape=(0, window, feature_n))
    train_y = np.empty(shape=(0, forecast_day))
    for i in range(len(train_data)-window-forecast_day+1):
        train_x = np.vstack((train_x, train_data[np.newaxis,i:(i+window),:]))
        train_y = np.vstack((train_y, train_data[(i+window):(i+window+forecast_day), 0]))

    train_x, validation_x = train_x[:int(len(train_x)*training_data_rate),:,:], train_x[int(len(train_x)*training_data_rate):,:,:] 
    train_y, validation_y = train_y[:int(len(train_y)*training_data_rate),:], train_y[int(len(train_y)*training_data_rate):,:]
    
return train_x, validation_x, train_y, validation_y, train, real_updown