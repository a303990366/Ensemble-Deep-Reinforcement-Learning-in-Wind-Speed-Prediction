import os
import numpy as np
import pandas as pd
import time
import math
import gym
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO,DDPG,A2C
from stable_baselines3.common.monitor import Monitor
from env import WeatherEnv
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class optimize_rl:
    def __init__(self,window_size,predict_days,total_timesteps):
        self.model = None
        self.window_size = window_size
        self.predict_days = predict_days
        self.total_timesteps = total_timesteps
        self.r2_list = []
        self.scaler = None

    def scaler_trans(self,data):
        if self.scaler is None:
            scaler = MinMaxScaler()
            norm_data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            norm_data = self.scaler.transform(data)
        return norm_data

    def train(self,data,label):
        original_data = data.copy()
        data = self.scaler_trans(data)
        env = Monitor(WeatherEnv(data, label, self.window_size,self.predict_days,original_data))
        model = A2C('MlpPolicy', env, verbose=1,device = 'cuda',tensorboard_log='./logs/rl')
        model.learn(total_timesteps=self.total_timesteps)
        self.model = model
        print('episode number: ',len(env.get_episode_rewards()))
        plt.plot(env.get_episode_rewards())
        prediction = self.validate(original_data,label)
        return prediction
    def validate(self,data, label):
        original_data = data.copy()
        data = self.scaler_trans(data)
        env = WeatherEnv(data, label, self.window_size,self.predict_days,original_data)
        model = self.model
        model.set_env(env)
        ob = env.reset()
        while True:
            action, _states = model.predict(ob)
            ob,reward,done,info = env.step(action)
            if done ==True:
                break
        self.show_r2(env.output_list, label)
        return env.output_list
    def test(self,data, label):
        return self.validate(data,label)
    def show_r2(self,prediction,label):
        print('Prediction shape: {0}, label shape: {1}'.format(prediction.shape,label.shape))
        r2 = r2_score(prediction,label[self.window_size-1:])
        self.r2_list.append(r2)
        print('r2 score : {0}'.format(r2))
    def main(self,X_train, y_train, X_test, y_test):
        print('Training the model...')
        prediction_train_drl = self.train(X_train, y_train)
        print('Testing the model...')
        prediction_test_drl = self.test(X_test, y_test) 
        return prediction_train_drl,prediction_test_drl