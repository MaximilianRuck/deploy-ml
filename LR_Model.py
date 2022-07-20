#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import pickle
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).parents[0]


data = pd.read_csv(ROOT / "Expenses_prediction.csv")



data.shape



scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(data)
df = pd.DataFrame.from_records(data_scaled)





X = df.iloc[:,:2]





Y= df.iloc[:,-1]






lr_model = LinearRegression()
lr_model.fit(X, Y)





Y_predict = lr_model.predict(X)
r2 = r2_score(Y, Y_predict)
print('R2 score is {}'.format(r2))



# Saving model to disk
pickle.dump(lr_model, open(ROOT / 'lr_model.pkl', 'wb'))







