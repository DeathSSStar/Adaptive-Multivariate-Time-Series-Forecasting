import pandas as pd
from functions import *
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (20,10)})
plt.rcParams.update({'axes.titlepad' : (20)})
plt.rcParams.update({'legend.fontsize' : (17)})
import os

def main(path,target_variable,prevision_frequency,prevision_length):

    df = pd.read_csv(path)

    if not(target_variable in list(df.columns)):
        print("Target variable "+target_variable+" not found in DataFrame")
        return

    if prevision_frequency == "hour":
        col_no = ["hour","day","month","year","day_of_week"]
        
    elif prevision_frequency == "day":
        col_no = ["day","month","year","day_of_week"]

    elif prevision_frequency == "month":
        col_no = ["month","year"]
        
    elif prevision_frequency == "year":
        col_no = ["year"]
    
    else:
        print("prevision_frequency: "+prevision_frequency+" is incorrect!")
        return

        
    # Cross Validation parameters based on dataframe length
    if len(df)>50000:
        windows = [30]
        batches = [256]
        patience = 3
        patience_final = 7

    elif len(df)>20000:
        windows = [48]
        batches = [64, 128, 256]
        patience = 5
        patience_final = 10
        
    elif len(df)>8000:
        windows = [18, 24, 30, 36, 48] 
        batches = [32, 64, 128]
        patience = 7
        patience_final = 15
        
    else:
        windows = [6, 12, 18, 24, 30, 36, 48]    
        batches = [16, 32, 64, 128]
        patience = 10    
        patience_final = 50

    #Cross Validation
    losses = cross_validation(df, windows, batches,col_no,target_variable,prevision_frequency,prevision_length,patience,epochs)

    #Find best batch and window_size
    index = np.argmin(losses)
    i=0
    for window_size in windows:
        for batch in batches:
            if i == index:
                best_batch = batch
                best_window = window_size
                break
            i+=1
    X_train, y_train, exog_train, X_test, y_test, exog_test, X_final, exog_final, yscaler, test_length = preprocessing(
        df,best_window,col_no,target_variable,prevision_frequency,prevision_length)

    model = create_model(X_train, y_train, exog_train)
    callback = tf.keras.callbacks.EarlyStopping(min_delta=0.0001,patience=patience_final,restore_best_weights=True)
    history = model.fit([X_train,exog_train],y_train, epochs=epochs, batch_size=batch,
            callbacks = [callback], validation_data = ([X_test,exog_test],y_test))
    
    plt.plot(history.history['loss'],label="Train")
    plt.plot(history.history['val_loss'],label="Test")
    plt.title('Model training',size=30)
    plt.ylabel('loss',size=17,rotation=0,labelpad=15)
    plt.xlabel('epoch',size=17)
    plt.legend()
    plt.show()

    y_pred = model.predict([X_test, exog_test])
    y_pred_scaled = yscaler.inverse_transform(y_pred)
    y_test_scaled = yscaler.inverse_transform(y_test.reshape(y_test.shape[0],y_test.shape[1]))
    MSE = np.sqrt(mse(y_pred_scaled[0],y_test_scaled[0]))

    y_forecast = model.predict([X_final, exog_final]).reshape(1,prevision_length)
    y_forecast_scaled = yscaler.inverse_transform(y_forecast)
    return MSE, y_forecast_scaled

if __name__=="__main__":
    filename = input("input dataset filename without extension ")
    path = "./datasets/"+filename+".csv"
    #path = os.path.join(os.getcwd(), "my_dir\\"+filename+".csv")
    target_variable = input("input target variable's name ")
    prevision_frequency = input("input time frequency, must be one of: [\"hour\",\"day\",\"month\",\"year\"] ")
    prevision_length = 36
    epochs = 100
    MSE, forecast = main(path,target_variable,prevision_frequency,prevision_length)
    print("MSE on the test set is :"+str(np.round(MSE,2))+"\n\n")
    print("The forecast is :"+str(forecast))
