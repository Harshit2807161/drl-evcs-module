# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:14:26 2024

@author: Harshit
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import fem
import classes 
from generate_uniform import generate_data
evcs1 = classes.EVCS(2,0.16)
evcs2 = classes.EVCS(2,0.16)
evcs3 = classes.EVCS(2,0.16)
class EVRL(gym.Env):

    def __init__(self,j):
        super().__init__()
        # Define action and observation space
        self.j = j
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = -np.inf, high= np.inf, shape = (19,), dtype=np.float64)
        self.charging_stations = [5, 22, 32]
        self.charging_stations_info = [evcs1,evcs2,evcs3]
        self.states = []
        self.actions = []
        self.i = 0 
        self.Map = classes.Map(39)
        self.time_mat = self.Map.get_time(39)
        
    def step(self, action):
        self.ev_requests = self.data
        charging_requests = len(self.data)-1
        self.ev = classes.EV(self.data.iloc[self.i,0],self.data.iloc[self.i,1],self.data.iloc[self.i,4],self.data.iloc[self.i,5], self.i)
        self.i = self.i + 1
        self.counter += self.data.iloc[self.i,3]
        if self.counter % 5 == 0 and self.counter!=0:
            self.time_mat = self.Map.get_time(39)
            self.counter = 0 
        self.ev.charging_station = action
        self.ev = classes.EV(self.data.iloc[self.i,0],self.data.iloc[self.i,1],self.data.iloc[self.i,4],self.data.iloc[self.i,5], self.i)
        if self.i == len(self.data)-1:
            self.terminated = True
            '''for i in range(charging_requests):
                summ = summ + fem.get_actual_costs(ev_requests[i],charging_stations,states[i],time_mat,Map.distance_matrix,charging_stations[actions[i]],actions[i])
                dist_sum = dist_sum + fem.get_total_distances(ev_requests[i],charging_stations,Map.distance_matrix,k=3)[actions[i]]
                time_sum = time_sum + fem.get_travel_times(ev_requests[i],charging_stations,states[i],Map.distance_matrix,k=3)[actions[i]]
                waiting_sum = waiting_sum + states[i][9+2]
                charging_sum = charging_sum + fem.get_charging_costs(ev_requests[i],charging_stations,states[i],Map.distance_matrix,k=3)[actions[i]]'''
        else:
            self.terminated = False
            
        self.next_state = fem.preprocess_data(self.ev,self.data.iloc[self.i,:].astype(int).tolist(), self.charging_stations, self.Map, self.time_mat,self.charging_stations_info, 3)
        self.states.append(self.next_state)
        self.actions.append(action)
        
        if self.terminated is True:
          ret_cost = [0]*charging_requests
          total = 0 
          for j in range(charging_requests):
              if self.data.iloc[j,6]==0:
                  ret_cost[j]=fem.get_actual_costs(self.ev_requests.iloc[j,:].astype(int).tolist(),self.charging_stations,self.states[j],self.time_mat,self.Map.distance_matrix,self.charging_stations[self.actions[j]],self.actions[j],k=3)-fem.get_travel_costs(self.ev_requests.iloc[j,:].astype(int).tolist(),self.charging_stations,self.states[j],self.Map.distance_matrix,action,k=3)[self.actions[j]]
              else:
                  ret_cost[j] = fem.get_actual_distance(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,self.actions[j],k=3)-fem.get_distance_cost(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,self.actions[j],k=3)
              total = total + ret_cost[j]
          self.reward = -total 
        else:
          return_reward = 0
          if self.data.iloc[self.i-1,6]==0:
              return_reward = fem.get_travel_costs(self.data.iloc[self.i-1,:].astype(int).tolist(),self.charging_stations,self.observation,self.Map.distance_matrix,action,k=3)[action]
          else:
              return_reward = fem.get_distance_cost(self.data.iloc[self.i-1,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,action,k=3)
          self.reward = -return_reward

        self.info = {}
        self.truncated = False
        self.observation = self.next_state
        return self.next_state, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None):
        self.data = generate_data(self.j)
        self.ev = classes.EV(self.data.iloc[0,0],self.data.iloc[0,1],self.data.iloc[0,4],self.data.iloc[0,5], 0)
        self.observation = fem.preprocess_data(self.ev,self.data.iloc[0,:].astype(int).tolist(), self.charging_stations, self.Map, self.time_mat,self.charging_stations_info, 3)
        self.states = [self.observation]
        self.actions = []
        self.info = {}
        self.i = 0
        self.time_mat = self.Map.get_time(39)
        self.counter = 0 
        return self.observation, self.info