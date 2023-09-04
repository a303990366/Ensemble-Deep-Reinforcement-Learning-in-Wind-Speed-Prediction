import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,GRU
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, MaxPooling1D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

import time
import pandas as pd

from processing_AQIdata import data_loader
import math
from sklearn.preprocessing import MinMaxScaler


class supervised_models:
    def __init__(self,epochs,shape):
        #parameter
        self.epochs = epochs
        self.shape = shape
        #data storation
        self.output_data_train = None
        self.output_data_val = None
        self.output_data_test = None
        self.r2_list = []
        #model
        self.model1 = Sequential([
                Conv1D(128, kernel_size=2, activation='relu', input_shape=self.shape,padding='same'),
                AveragePooling1D(pool_size=2,padding='same'),
                GRU(64),
                # Conv1D(128, kernel_size=2, activation='relu',padding='same'),
                # AveragePooling1D(pool_size=2,padding='same'),
                #TCN(nb_filters = 128,kernel_size=2,padding='same'),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1)
            ])
        self.model2 = Sequential([
                LSTM(64,return_sequences=True ,input_shape=self.shape),
                LSTM(64),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1)
            ])
        self.model3 = Sequential([
                Conv1D(128,kernel_size = 2, activation='relu',input_shape=self.shape,
                	padding = 'same'),
                AveragePooling1D(pool_size=2,padding='same'),
                # Conv1D(128,kernel_size = 2, activation='relu',padding = 'same'),
                # AveragePooling1D(pool_size=2,padding='same'),
                LSTM(64),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1)
            ])
        self.model_list = [self.model1,self.model2,self.model3]
        self.history_list = []
        self.optimizer = Adam(learning_rate=0.0005)
    def train(self,X_train,y_train):
        for model_name in range(len(self.model_list)):
            model = self.model_list[model_name]
            model.compile(optimizer=self.optimizer, loss='mse', metrics=[MeanSquaredError(),MeanAbsoluteError()])
            # Train the model
            tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/{0}".format(model_name))
            history = model.fit(X_train, y_train, epochs=self.epochs,verbose=0,callbacks=[tf_callback])
            self.history_list.append(history)
            self.model_list[model_name] = model 
                        
            history = self.history_list[model_name]
            
            # plt.plot(history.history['loss'],label='train')
            # plt.plot(history.history['val_loss'],label = 'validate')
            # plt.title('model {}'.format(model_name))
            # plt.legend()
            # plt.show()
    def evaluate(self,X,y):
        for model_name in range(len(self.model_list)):
            model = self.model_list[model_name]
            predictions = model.predict(X,verbose=0)
            # Evaluate the model 
            mse = model.evaluate(X, y,verbose=0)[1] # it's the same as loss function
            mae = model.evaluate(X, y,verbose=0)[2]
            r2 = r2_score(y, predictions)
            self.r2_list.append(r2)
            print('model: {}'.format(model_name))
            print('Mean Squared Error:', mse)
            print('Mean Absolute Error:', mae)
            print('R2 Score:', r2)

    def predict(self,X,y):
        predict_list = []
        for model_name in range(len(self.model_list)):
            model = self.model_list[model_name]
            pred_y = model.predict(X,verbose=0).reshape(-1,)
            predict_list.append(pred_y)
        df = pd.DataFrame(predict_list).T
        return df
    def main(self,X_train,y_train,X_test,y_test):
        self.train(X_train,y_train)
        print('Testing...')
        self.evaluate(X_test,y_test)
        self.output_data_train = self.predict(X_train,y_train)
        self.output_data_test = self.predict(X_test,y_test)