# V3 Update:
#1. combine validation set and testing set for results
#2. add more stations to test

#!pip install ewtpy
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import numpy as np
import ewtpy
import time
import pandas as pd
from processing_AQIdata import data_loader
import matplotlib.pyplot as plt
import math
from basic_models import supervised_models
from env import WeatherEnv
from drl import optimize_rl

def create_sequences(data, target, sequence_length,head):
    X, y = [], []
    for i in range(len(data) - sequence_length-head):
        X.append(data[i:(i + sequence_length)])
        y.append(target[(i-1+head) + sequence_length])
    return np.array(X), np.array(y)


if __name__=="__main__":
    ori_path = r"C:\Users\x5748\Downloads\研究所\實驗室\Shahab\wind prediction\壓縮檔\北部空品區"
    input_features_list = [
    ['WS_HR'],
    # ['WS_HR','diff_ma'],
    # ['WS_HR','MovingAverage'],
    # ['WS_HR','diff_ma','MovingAverage']
    ]
    # Define the sequence length
    #sequence_length = 6
    head = 1
    calc_times = [i for i in range(2,8)]
    for place in ['土城','士林','板橋','富貴角','基隆']:#'土城','士林','板橋','富貴角',
        final_results = pd.DataFrame()
        for sequence_length in calc_times:
            print('Doing an experiment for Sequence_length: {0},place: {1}'.format(sequence_length,place))
            '''
            AMB_TEMP 溫度
            RAINFALL 雨量
            '''
            rl_r2_store = []
            store_r2 = []
            start = time.time()
            for input_features in input_features_list:
                loader = data_loader()
                df = loader.main(ori_path,place,2018,2022,freq='D')
                #df = pd.read_csv(r"C:\Users\x5748\Downloads\研究所\實驗室\Shahab\wind prediction\wind_data_example.csv")
                #df['Date'] = pd.to_datetime(df['Date'])
                #df = df.set_index('Date')
                #df = df.asfreq('D')
                #df = df.interpolate()
                df['diff'] = df['WS_HR'] / df['WIND_SPEED'] 
                #add
                window_size = 7  # Define the window size for the moving average
                df['MovingAverage'] = df['WS_HR'].rolling(window_size).mean()
                df['RAINFALL'] = df['RAINFALL'].rolling(window_size).mean()
                df['AMB_TEMP'] = df['AMB_TEMP'].rolling(window_size).mean()
                df['WD_HR'] = df['WD_HR'].rolling(window_size).mean()
                df = df.iloc[window_size:,:]
                df['diff_ma'] = df['WS_HR'] / df['MovingAverage']

                original_df = df
                label_df = df[['WS_HR']]
                df = df[input_features]
                #df = df[['WS_HR','diff_ma','MovingAverage']],
                #diff_ma, moveingAverage is useful
                print('data shape before decomposed: ',df.shape)

                processed_data = None
                for col in range(0,df.shape[1]):
                    tmp = df.iloc[:,col]
                    ewt,  mfb ,boundaries = ewtpy.EWT1D(tmp, N = 100) #original N = 40
                    if col == 0:
                        processed_data = ewt
                    else:
                        processed_data = np.hstack((processed_data,ewt))
                print(input_features,'processed_data shape: ',processed_data.shape)

                X = processed_data
                y = label_df.to_numpy()#df[["WS_HR"]].to_numpy()

                # Split the data into training, validation, and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)
                # Apply Min-Max normalization
                scaler = MinMaxScaler()
                X_train= scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)


    		    # Create sequences for training, validation, and testing sets
                X_train, y_train = create_sequences(X_train, y_train, sequence_length,head)
                X_test, y_test = create_sequences(X_test, y_test, sequence_length,head)
                print(X_train.shape,y_train.shape)

                #train model
                basic_models = supervised_models(epochs = 40,shape = (sequence_length, X_train.shape[2]))

                basic_models.main(X_train,y_train,X_test,y_test)

                # store_r2.append(basic_models.r2_list)
                window_size = 7
                predict_days = 0
                total_timesteps = 20000
                main_rl = optimize_rl(window_size,predict_days,total_timesteps)
                prediction_train_drl,prediction_test_drl = main_rl.main(basic_models.output_data_train, pd.DataFrame(y_train) ,
                         basic_models.output_data_test, pd.DataFrame(y_test))

                basic_models.r2_list.append(main_rl.r2_list[1])
                store_r2.append(basic_models.r2_list)
                #store_r2.append(basic_models.r2_list+ main_rl.r2_list[1])

                #add
                output_train = pd.concat([basic_models.output_data_train, pd.DataFrame(y_train),pd.DataFrame(prediction_train_drl)],axis = 1)
                output_test = pd.concat([basic_models.output_data_test, pd.DataFrame(y_test),pd.DataFrame(prediction_test_drl)],axis = 1)
                columns_output = ['cnn-gru','2lstm','cnn-lstm','label','DRL']
                output_train.columns = columns_output
                output_test.columns = columns_output

                train_save_place = "./results/place/{0}/train".format(place)
                test_save_place = "./results/place/{0}/test".format(place)

                if not os.path.isdir(train_save_place):
                    os.makedirs(train_save_place)
                if not os.path.isdir(test_save_place):
                    os.makedirs(test_save_place)
                output_train.to_csv(os.path.join(train_save_place,'prediction_with_label_train_{0}_{1}.csv'.format(place,sequence_length)),index=False)
                output_test.to_csv(os.path.join(test_save_place,'prediction_with_label_test_{0}_{1}.csv'.format(place,sequence_length)),index=False)


            end = time.time()
            print('Time cost: ',end-start)

            tpp = pd.DataFrame(store_r2,columns = ['cnn_gru_test','lstm_test','cnn_lstm_test','DRL_test'],
                    index = [','.join(i) for i in input_features_list])
            tpp['overall_basic_test'] = tpp.iloc[:,0:3].mean(axis = 1)
            tpp = tpp.round(3)
            final_results = pd.concat([final_results,tpp])
        print(final_results)
        final_results.index = calc_times
        final_results.to_csv('C:/Users/x5748/Downloads/研究所/實驗室/Shahab/wind prediction/results/results{0}_V3.csv'.format(place),index = True)

        

