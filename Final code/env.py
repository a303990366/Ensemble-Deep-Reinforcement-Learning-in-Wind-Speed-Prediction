import numpy as np
import pandas as pd
from processing_AQIdata import data_loader
import math
import gym
from gym import spaces
from gym.utils import seeding


class WeatherEnv(gym.Env):
    def __init__(self, data, label, window_size,predict_days,original_data):
        '''
        window_size: the length of data
        predict_days: How many future day we wanna predict
        '''
        #assert data.ndim == 2
        self.base_model_num = 3
        self.window_size = window_size
        self.label, self.signal_features = label, data
        self.original_data = original_data
        self.shape = (window_size, self.signal_features.shape[1])
        self.predict_days = predict_days
        # spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(self.base_model_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.shape[0]*self.shape[1],), dtype=np.float32)
        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.label) - 1
        self._done = None
        self._current_tick = None
        self._total_reward = None
        self.reward_list = []
        self.action_list = []  
        self.output_list = []
        self.results = None

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._total_reward = 0
        self.reward_list = []
        self.action_list = []
        self.results = None
        self.output_list = []
        return self._get_observation()
    def plot_cumulative_reward(self):
        plt.plot(np.cumsum(self.reward_list))
        plt.show()
    def step(self, action):
        self._done = False
        self._current_tick += 1

        step_reward = self._calculate_reward(action)
        self.reward_list.append(step_reward)
        observation = self._get_observation()
        
        if self._done == True:
#             self.combine_label_prediction()
            self.output_list = np.array(self.output_list)

        return observation, step_reward, self._done, {}

    def _get_observation(self):
        ob = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
        # add variance as ob 
        ob = ob.flatten()
        return ob

    def _calculate_reward(self, action):
        action = self.softmax_normalization(action)
        self.action_list.append(action)
        #label -prediction
        pre_tick = self._current_tick - 1 + (self.predict_days - 1)
        lab = self.label.iloc[pre_tick].values[0]
        data = self.original_data.iloc[pre_tick].values
        #print(data)
        if pre_tick == self._end_tick :
            self._done = True
        
        output = np.dot(data,action)
        self.output_list.append(output)
        diff = output - lab
        if np.abs(diff) < 0.15:
            reward = 1
        else:
            reward = 0
        #reward = -(diff**2)+0.1
#         reward = np.clip(reward,-1,1)
        return reward
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output