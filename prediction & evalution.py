# Make 7 days arraies for evalution
# (1) value of prediction (2) actual value (3) prediction of trend (4) actual trend (5) weather prediction of trend is accurate or not

def pred_7d(model_saved, validation_x, validation_y, forecast_day, scaler, train, real_updown):
    
    model = load_model(model_saved)
    validation_set_pred = model.predict(validation_x, batch_size=4)

    if scaler == MinMaxScaler():
        validation_set_pred = validation_set_pred * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]
        validation_y = validation_y * (scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0]
        
    elif scaler == StandardScaler():
        validation_set_pred = validation_set_pred * (scaler.var_[3]**(1/2)) + scaler.mean_[3]
        validation_y = validation_y * (scaler.var_[3]**(1/2)) + scaler.mean_[3]
    
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
        
    return predictions  

# Evalute the accuracy of the model
def show_evaluation(predictions,forecast_day):
    for i in range(forecast_day):
        print("Trend accuracy of", i+1,"days after：", predictions[:,i,4].mean())
    print("Model: " + model_saved + " average trend accuracy：", sum([predictions[i,:,4].mean() for i in range(len(predictions))])/len(predictions))

# Make dataframe of the accuracy of the model, prepare for multi-model/weight ensembling
def df_evaluation(predictions,forecast_day):
    eval_dt = np.zeros(shape=(forecast_day+1, 1))
    index = np.zeros(shape=(forecast_day+1, 1))
    
    # evaluate everyday trend accuracy
    for i in range(forecast_day):
        eval_dt[i,0]=predictions[:, i, 4].mean()
    
    # evaluate average trend accuracy
    eval_dt[-1, 0] = sum([predictions[i, :, 4].mean() for i in range(len(predictions))])/len(predictions)
    
    index = ['%s %s'%(n, 'Day') for n in list(range(1, i+2))]
    index.append('Average')

    df = pd.DataFrame(eval_dt, columns = [model_saved])
    df.index = index
    return df