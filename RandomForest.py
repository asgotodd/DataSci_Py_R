import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
import sys # import sys to get more detailed Python exception info
import psycopg2 # import the connect library for psycopg2
from psycopg2 import OperationalError, errorcodes, errors # import the error handling libraries for psycopg2
import psycopg2.extras as extras
from sqlalchemy import create_engine
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import numpy as np
from sqlalchemy import create_engine
import json
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.dialects import registry
import datetime


##Create Random Forest Model
rf_iter=1
symb_test=[]
col_list = ['symbol','prediction_type','prediction']
prediction_values = pd.DataFrame(columns=col_list)


#DataFrame for "RF_Data.CSV" - Remove Symbol (non-numeric) first column
RF_data_all=pd.read_csv(#<local path> "RF_Data.CSV" )
RF_data=RF_data_all.iloc[:, 1:]

#Separate Columns
X_train = RF_data.iloc[:, :-1].values
y = RF_data.iloc[:, -1].values


#Build Random Forest - 1,000 trees
# Runtime with no scaler - 15ish mins
rf = RandomForestRegressor(n_estimators=1000)
rf.fit(X_train, y)




##Iterate for 1 Prediction for 1 Symbol - Local File with Symbol - RF_SYMBOL.CSV
while rf_iter < 2:
    symb_test=pd.read_csv(#<Local Path> "RF_Symbol.CSV" )
    if len(symb_test) != 1:
        rf_iter += 1
        continue
    else:
#import Random Forest (rf) data from "RF_Data.CSV" - Remove Symbol (non-numeric) first column
        RF_pred_all=pd.read_csv(#<local path> "RF_Data.CSV" )
        RF_pred=RF_pred_all.iloc[:, 1:]
        X_train = RF_pred.iloc[:, :-1].values
#Loop for 1 prediction
        pred_iter=1
        while pred_iter > 0:
            prediction_values.loc[len(prediction_values)]=[symb_test['symbol'][0],'RandomForest',rf.predict(X_train)[0]]
            pred_iter -= 1
        rf_iter += 1

