from __future__ import print_function
import pandas as pd #The Pandas data science library
import requests #The requests library for HTTP requests in Python
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from iexfinance.refdata import get_symbols
import sys # import sys to get more detailed Python exception info
import psycopg2 # import the connect library for psycopg2
from psycopg2 import OperationalError, errorcodes, errors # import the error handling libraries for psycopg2
import psycopg2.extras as extras
from sqlalchemy import create_engine
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import numpy as np
from sqlalchemy import create_engine
import json

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Flatten, GRU, LSTM, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException



rnn_iter=1
symb_test=[]
col_list = ['symbol','prediction_type','prediction']
prediction_values = pd.DataFrame(columns=col_list)


#Load Sample File "RNN_LSTM_Symbol.CSV" to DataFrame
while rnn_iter < 1:
    symb_test=pd.read_csv(#<local path> 'RNN_LSTM_Symbol.CSV')
    if len(symb_test) != 1:
        rnn_iter += 1
        continue
    else:
#import data from CSV Sample File - "RNN_LSTM_Data.CSV."
        df=pd.read_csv(#<local path> 'RNN_LSTM_Data.CSV')
#Calculate the % return by shifting data (Prev Close)
        df['PrevClose']=df['close'].shift(1)
#Calculate the % return
        df['Return']=(df['close']-df['PrevClose'])/df['PrevClose']
        df.dropna(subset=['Return'],inplace=True)
#Assign input and targets to numpy array
        input_data=df[['fact1','fact3','fact3','fact4','fact5']].values
        targets=df['Return'].values
        T=10 #Number of time-periods back to use in prediction
        D=input_data.shape[1]
        N=len(input_data)-T
#Normalize the data - Train = all - T
        Ntrain=len(input_data)-T
        scaler=StandardScaler()
        scaler.fit(input_data[:])
        input_data=scaler.transform(input_data)     
#Setup XTrain and YTrain
        X_train=np.zeros((Ntrain,T,D))
        Y_train=np.zeros(Ntrain)
        for t in range(Ntrain):
            X_train[t,:,:]=input_data[t:t+T]
            Y_train[t]=(targets[t+T]>0)
#Setup X_test and Y_test
        X_test=np.zeros((N-Ntrain,T,D))
        Y_test=np.zeros(N-Ntrain)
        for u in range(N-Ntrain):
        #u counts from 0 to N-Ntrain
        #t counts from Ntrain to N
            t=u+Ntrain
            X_test[u,:,:]=input_data[t:t+T]
            Y_test[u]=(targets[t+T]>0)
#RNN
        i = Input(shape=(T,D))
        x=SimpleRNN(50)(i) #Default Activation is TAN-H
        x=Dense(1,activation='sigmoid')(x)
        model=Model(i,x)
        model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001)
        )
#train the model
        r=model.fit(
        X_train,Y_train,
        batch_size=8192,
        epochs=250
        )
#Loop for 5 predictions - DF 'prediction_values' holds next 5 day's predictions
        pred_iter=5
        while pred_iter > 0:
            prediction_values.loc[len(prediction_values)]=[df['symbol'][1],'RNN',model.predict(X_train[Ntrain-pred_iter:Ntrain-pred_iter+1])[0][0]]
            pred_iter -= 1
#Assign input and targets to numpy array
        input_data=df[['fact1','fact3','fact3','fact4','fact5']].values
        targets=df['Return'].values
        T=10 #Number of time-periods back to use in prediction
        D=input_data.shape[1]
        N=len(input_data)-T
#Normalize the data - Train = all - T
        Ntrain=len(input_data)-T
        scaler=StandardScaler()
        scaler.fit(input_data[:])
        input_data=scaler.transform(input_data)     
#Setup XTrain and YTrain
        X_train=np.zeros((Ntrain,T,D))
        Y_train=np.zeros(Ntrain)
        for t in range(Ntrain):
            X_train[t,:,:]=input_data[t:t+T]
            Y_train[t]=(targets[t+T]>0)
#Setup X_test and Y_test
        X_test=np.zeros((N-Ntrain,T,D))
        Y_test=np.zeros(N-Ntrain)
        for u in range(N-Ntrain):
        #u counts from 0 to N-Ntrain
        #t counts from Ntrain to N
            t=u+Ntrain
            X_test[u,:,:]=input_data[t:t+T]
            Y_test[u]=(targets[t+T]>0)
#LSTM
        i = Input(shape=(T,D))
        x=LSTM(50)(i) #Default Activation is TAN-H
        x=Dense(1,activation='sigmoid')(x)
        model=Model(i,x)
        model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001)
        )
#train the model
        r=model.fit(
        X_train,Y_train,
        batch_size=8192,
        epochs=250
        )
#Loop for 5 predictions - DF 'prediction_values' holds next 5 day's predictions
        pred_iter=5
        while pred_iter > 0:
            prediction_values.loc[len(prediction_values)]=[df['symbol'][1],'LSTM',model.predict(X_train[Ntrain-pred_iter:Ntrain-pred_iter+1])[0][0]]
            pred_iter -= 1
