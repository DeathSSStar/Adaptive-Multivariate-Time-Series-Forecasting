from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Flatten, Dense, Bidirectional, Input, Concatenate  


def encode_non_numerical(df):
    """
    Questa funzione controlla se ci sono colonne non numeriche
    e in caso ci siano le codifica con il label encoder.
    
    Da eseguire dopo la funzione missing data
    """
    columns = list(df.columns)
    columns.remove("datetime")
    for col in columns:
        if not(df[col].dtype.kind in "iuf"):
            le = OneHotEncoder()
            encoded = le.fit_transform(df[[col]]).toarray()
            for i in range(encoded.shape[1]):
                df[col+"_"+str(i)] = encoded[:,i]
            df.drop([col],axis=1,inplace=True)
    
    return df



def missing_data(df):
    if df.isnull().sum().sum() != 0:
        nulls = df.isnull().sum()
        columns_null = list(nulls[nulls!=0].index)
        for col in columns_null:
            df[col] = df[col].interpolate(option="polynomial",order=5)
    else:
        print("No missing data!")
    return 


def time_df(df, prev_freq = "day"):
    """
    Questa funzione verrà usata per generare le colonne relative
    a data e ora per il train e per controllare se datetime è nel formato corretto.
    
    prev_freq sarà la frequenza delle previsioni ["hour","day","month","year"]
    """
    # dato che non so da che giorno cominci il dataset, faccio conto che inizi dal
    # caso peggiore, ovvero il primo giorno del mese e così mi occore fare 13 controlli.
    if prev_freq == "hour":
        for i in range(13):
            temp = df["datetime"][i*24]
            temp = int(temp.split("-")[1])
            if temp > 12:
                print("Errore in datetime. Prego inserire un dataframe nel formato corretto")

    if prev_freq == "day":
        for i in range(13):
            temp = df["datetime"][i]
            temp = int(temp.split("-")[1])
            if temp > 12:
                print("Errore in datetime. Prego inserire un dataframe nel formato corretto")

    if prev_freq == "month":
        for i in range(13):
            temp = df["datetime"][i]
            temp = int(temp.split("-")[1])
            if temp > 12:
                print("Errore in datetime. Prego inserire un dataframe nel formato corretto")
    
    if prev_freq == "year":
        temp = df["datetime"][0]
        temp = int(temp.split("-")[1])
        if temp > 12:
            print("Errore in datetime. Prego inserire un dataframe nel formato corretto")
    
    df["datetime"] = pd.to_datetime(df["datetime"],dayfirst=True)
    
    if prev_freq == "hour":
        df["hour"] = df["datetime"].dt.strftime("%H")
        df["day"] = df["datetime"].dt.strftime("%d")
        df["month"] = df["datetime"].dt.strftime("%m")
        df["year"] = df["datetime"].dt.strftime("%Y")
        df["day_of_week"] = df["datetime"].dt.dayofweek
        
        
    elif prev_freq == "day":
        df["day"] = df["datetime"].dt.strftime("%d")
        df["month"] = df["datetime"].dt.strftime("%m")
        df["year"] = df["datetime"].dt.strftime("%Y")
        df["day_of_week"] = df["datetime"].dt.dayofweek
        
    elif prev_freq == "month":
        df["month"] = df["datetime"].dt.strftime("%m")
        df["year"] = df["datetime"].dt.strftime("%Y")
        
    elif prev_freq == "year":
        df["year"] = df["datetime"].dt.strftime("%Y")
    
    return df


def create_step(df, num_steps):
    """
    Questa funzione prende il generico dataframe a singola
    variabile e crea tante colonne shiftate
    quanti sono i num_steps. l'operazione non viene 
    eseguita inplace.
    
    df deve essere un dataframe con una singola variabile.
    """
    df1 = pd.DataFrame()
    column = list(df.columns)[0]
    for step in range(num_steps):
        col_name = column + "_" + str(step+1)
        df1[col_name] = df[column].shift(step+1).fillna(0)
            
    return df1


def shift_all(df, columns_no = [], num_steps = 1):
    """
    Questa funzione applica create step a tutte le colonne tranne
    alla variabile di target e quelle presenti in columns_no, che
    di default sono le colonne relative a data ed ora.
    
    nun_steps : stabilisce quanti shift verranno applicati alle variabili
    columns_no : lista di nomi colonne a cui non verrà applicato lo shift
    """
    columns_ = list(df.columns)
    columns  = [x for x in columns_ if x not in columns_no]
    df_final = df[columns_no]
    
    for col in columns:
        temp = df[col].to_frame()
        temp2 = create_step(temp, num_steps)
        df_final = pd.concat([df_final, temp2], axis=1)
    
    return df_final


def preprocessing(df_,window_size,col_no,target_variable,prevision_frequency,prevision_length):
    df = df_.copy()
    missing_data(df)
    df = encode_non_numerical(df)
    
    num_col = len(list(df.columns))
    while True:
        if window_size*(num_col+len(col_no)-2) > len(df):
            window_size = int(window_size/2)
            print("Dataset too small! \nwindow_size reduced to: ",window_size)
        else:
            print("\n --- \n")
            break
    
    
    df = time_df(df, prevision_frequency)
    
    yscaler = MinMaxScaler()
    y_ = np.array(df[target_variable]).reshape(-1,1)
    y_ = yscaler.fit_transform(y_)
    df_target_variable = pd.DataFrame(y_,columns=[target_variable])    
    
    df = df.drop([target_variable,"datetime"],axis=1)
    
    exogscaler = MinMaxScaler()
    exog = exogscaler.fit_transform(df)
    columns = list(df.columns)
    df = pd.DataFrame(exog,columns=columns)
    
    df = shift_all(df, columns_no = col_no, num_steps = window_size)
    exog = df.to_numpy( )
    
    pca = PCA(exog.shape[1])
    pca = pca.fit(exog)
    variance = pca.explained_variance_ratio_
    variance_threshold = 0.8
    check = False
    max_comp = 5
    if max_comp > exog.shape[1]:
        max_comp = exog.shape[1]

    while True:
        app = 0
        for i in range(max_comp):
            app += variance[i]
            if app > variance_threshold:
                index=i+1
                check = True
                break

        if check == True:
            break

        elif variance_threshold < 0.61:
            index=max_comp
            break

        else:
            variance_threshold -= 0.1

    exog_pca = pca.transform(exog)[:,:index]
    
    df_target_variable_ = create_step(df_target_variable,window_size)        
    df1 = pd.concat([df_target_variable,df_target_variable_],axis=1)
    X = df1.to_numpy()
    
    test_length = int(X.shape[0]/20) # 5% of dataset for the test set

    if test_length < prevision_length:
        test_length = prevision_length
    
    y = []
    for i in range(window_size,len(y_)-prevision_length):
        y.append(y_[i:i+prevision_length,0])
    y = np.array(y)
    
    X_train = X[window_size:-prevision_length-test_length]
    X_test  = X[-prevision_length-test_length:-prevision_length]
    X_final = X[-1].reshape(1,X.shape[1],1)
    
    y_train = y[:-test_length]
    y_test = y[-test_length:]
    
    exog_train = exog_pca[window_size:-prevision_length-test_length]
    exog_test = exog_pca[-prevision_length-test_length:-prevision_length]
    exog_final = exog_pca[-1].reshape(1,-1)
    
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],1)
    y_test = y_test.reshape(y_test.shape[0],y_test.shape[1],1)
    
    return X_train, y_train, exog_train, X_test, y_test, exog_test, X_final, exog_final, yscaler, test_length


def create_model(X_train, y_train, exog_train):
    X_input = Input(shape=(X_train.shape[1],1))
    exog_input = Input(shape=(exog_train.shape[1]))

    Bid_0 = Bidirectional(LSTM(X_train.shape[1], activation='tanh', return_sequences=True))(X_input) 
    Bid_1 = Bidirectional(LSTM(X_train.shape[1], activation='tanh', return_sequences=True))(Bid_0)
    Bid_2 = Bidirectional(LSTM(X_train.shape[1], activation='tanh', return_sequences=True))(Bid_1)
    drop_out_0 = Dropout(0.3)(Bid_2)
    flat = Flatten()(drop_out_0)

    concat = Concatenate()([flat,exog_input])
    dense_0 = Dense(1000, activation="tanh")(concat)
    drop_out_1 = Dropout(0.3)(dense_0)
    dense_3 = Dense(y_train.shape[1])(drop_out_1)

    model = Model([X_input,exog_input],dense_3)
    model.compile(optimizer="Adam", loss='mse')
    
    return model


def cross_validation(df, windows, batches, col_no,target_variable,prevision_frequency,prevision_length,patience,epochs):
    losses = []
    for window_size in windows:
        for batch in batches:
            X_train, y_train, exog_train, X_test, y_test, exog_test, X_final, exog_final, yscaler, test_length = preprocessing(
                df,window_size,col_no,target_variable,prevision_frequency,prevision_length)

            model = create_model(X_train, y_train, exog_train)
            callback = tf.keras.callbacks.EarlyStopping(min_delta=0.0001,patience=patience,restore_best_weights=True)
            history = model.fit([X_train,exog_train],y_train, epochs=epochs, batch_size=batch,
                        callbacks = [callback], validation_data = ([X_test,exog_test],y_test))
            
            val_loss = min(history.history["val_loss"])
            losses.append(val_loss)
            
    return losses
