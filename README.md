# Adaptive-Multivariate-Time-Series-Forecasting

This Repository is the code of my Master's thesis work.
This is going to be deployed on LasDigitalBrain, a platform developed by FLUEL s.r.l.s. that offers microservices with an API interface.

The execution requires 3 inputs via shell: 
- the dataset name (without .csv extention)
- the target variable's name
- the frequency of the data ("hour","day","month","year")

If you want to use it there are the 3 dataset that i used in the folder /datasets .
However if you want to use your own dataset you just need to paste it in the /datasets folder on your computer.

---
## Preprocessing
The preprocessing is made with the following steps:
- Check missing data
- Check non numerical columns
- Check datetime correct format
- Extract time info

---
For the target variable are executed the following steps:
- Rescale in range [0,1]
- Creation of the X array with *window_size* shifts in the past for training
- Creation of the y array with *prevision_length* future time shifts

---
For the exogenous variable are executed the following steps:
- Rescale in range [0,1]
- Creation of the *window_size* past time shifts
- Principal Component Analysis with automated choiche of N_components (exog_pca)

---
The final steps are:
- split X, y and exog_pca in 95% train and 5% test
- Reshape X and y with shape [Dim0, Dim1, 1]

![alt text](https://github.com/DeathSSStar/Adaptive-Multivariate-Time-Series-Forecasting/blob/master/Preprocessing%20tesi3.png)

---
## Bidirectional LSTM
The model is a Bidirectional LSTM with the following structure:

![alt text](https://github.com/DeathSSStar/Adaptive-Multivariate-Time-Series-Forecasting/blob/master/BD%20LSTM%20schema.png)

## Output

The outputs of the model are the MSE on the test set for the best model and the forecast.
Moreover it shows a plot with the validation curves, e.g.

![alt text](https://github.com/DeathSSStar/Adaptive-Multivariate-Time-Series-Forecasting/blob/master/LSTM_Gold_val_curve.png)
