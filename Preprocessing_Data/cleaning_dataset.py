import numpy as numpy
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("insurance_data.csv")
#print(df.head())

from sklearn.model_selection  import train_test_split
X_train, X_test , y_train , y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,test_size=0.2,random_state=25)

#print(len(X_train))    
#print(len(X_test))

X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age']/100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100

