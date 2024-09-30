import gymnasium as gym
import numpy as np
from gymnasium import spaces
import fem
from collections import deque
import classes 
from generate_uniform import generate_data
class EVRL(gym.Env):

    def __init__(self,j,dist):
        super().__init__()
        # Define action and observation space
        self.j = j
        self.dist = dist
        self.action_space = spaces.Discrete(3)
        self.evcs1 = classes.EVCS(2,0.16)
        self.evcs2 = classes.EVCS(2,0.16)
        self.evcs3 = classes.EVCS(2,0.16)
        self.observation_space = spaces.Box(low = -np.inf, high= np.inf, shape = (19,), dtype=np.float64)
        self.charging_stations = [5, 22, 32]
        self.charging_stations_info = [self.evcs1,self.evcs2,self.evcs3]
        self.states = []
        self.actions = []
        self.i = 0 
        self.Map = classes.Map(39)
        self.time_mat = self.Map.get_time(39)
        self.evs = []
        self.scores = []
        self.scores_0 = []
        self.scores_1 = []                       
        self.total_episodic_distances = []
        self.total_episodic_times = []
        self.total_episodic_waiting_times = []
        self.total_episodic_charging_times = []
        self.total_actual_travel_costs = []
        self.agent_2 = []
        self.scores_window = deque(maxlen=25) 
        self.dist_window = deque(maxlen=25)
        self.time_window = deque(maxlen=25)
        self.waiting_window = deque(maxlen=25)
        self.charging_window = deque(maxlen=25)
        self.actual_window = deque(maxlen=25)
        self.agent_2_dq = deque(maxlen=25)
        self.score_0_dq = deque(maxlen=25)
        self.score_1_dq = deque(maxlen=25)
        self.avg_episodic_rew = []
        self.avg_distance = []
        self.avg_time = []
        self.avg_charging_time = []
        self.avg_waiting_time = []
        self.avg_actual_travel_cost = []
        self.avg_agent_2 = []
        self.avg_score_0 = []
        self.avg_score_1 = []
        self.summ = 0
        self.dist_sum = 0 
        self.time_sum = 0
        self.waiting_sum = 0
        self.charging_sum = 0
        self.score = 0
        self.score_0 = 0
        self.score_1 = 0
        self.actual_agent_2 = 0
        self.charging_requests=0
    def step(self, action):
        self.ev_requests = self.data
        self.charging_requests = len(self.data)-1
        self.i = self.i + 1
        self.counter += self.data.iloc[self.i,3]
        if self.counter % 5 == 0 and self.counter!=0:
            self.time_mat = self.Map.get_time(39)
            self.counter = 0 
        self.ev = classes.EV(self.data.iloc[self.i-1,0],self.data.iloc[self.i-1,1],self.data.iloc[self.i-1,4],self.data.iloc[self.i-1,5], self.i-1)
        
        if self.i == len(self.data)-1:
            self.terminated = True
        else:
            self.terminated = False
        
        print("The starting node: ", self.data.iloc[self.i-1,0])
        print("The charging station chosen: ", self.charging_stations[action])
        print("The last node: ", self.data.iloc[self.i-1,1])
        if self.data.iloc[self.i-1,6]==0:
            print("The path taken: ", fem.djikstra_with_path(self.time_mat,self.data.iloc[self.i-1,0],self.charging_stations[action])[1] + fem.djikstra_with_path(self.time_mat,self.charging_stations[action],self.data.iloc[self.i-1,1])[1])
            print("\n")
        else:
            print("The path taken: ", fem.djikstra_with_path(self.time_mat,self.data.iloc[self.i-1,0],self.charging_stations[action])[1] + fem.djikstra_with_path(self.Map.distance_matrix,self.charging_stations[action],self.data.iloc[self.i-1,1])[1])
            print("\n")
        self.ev.charging_station = action
        self.charging_stations_info[action].confirm_reservation(self.ev, action, self.i-1)
        self.ev.charging_done_time = (self.i-1) + fem.get_driving_distances(self.ev.position, self.charging_stations, self.time_mat, 3)[action] + fem.get_expected_waiting_times(self.ev, self.charging_stations_info)[action] + self.charging_stations_info[action].charging_time(self.ev)
        for self.ev in self.evs:
            if (self.i-1) == self.ev.charging_done_time:
                print('yes')
                self.charging_stations_info[self.ev.charging_station].remove_ev(self.ev)
        self.ev = classes.EV(self.data.iloc[self.i,0],self.data.iloc[self.i,1],self.data.iloc[self.i,4],self.data.iloc[self.i,5], self.i)
        self.next_state = fem.preprocess_data(self.ev,self.data.iloc[self.i,:].astype(int).tolist(), self.charging_stations, self.Map, self.time_mat,self.charging_stations_info, 3)
        self.states.append(self.next_state)
        self.actions.append(action)
        self.evs.append(self.ev)
        
        if self.terminated is True:
          for j in range(self.charging_requests):
              if self.data.iloc[j,6]==0:
                  self.summ = self.summ + fem.get_actual_costs(self.ev_requests.iloc[j,:].astype(int).tolist(),self.charging_stations,self.states[j],self.time_mat,self.Map.distance_matrix,self.charging_stations[self.actions[j]],self.actions[j],k=3)
                  self.dist_sum = self.dist_sum + fem.get_total_distances(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,k=3)[self.actions[j]]
                  self.time_sum = self.time_sum + fem.get_travel_times(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.states[j],self.Map.distance_matrix,k=3)[self.actions[j]]
                  self.waiting_sum = self.waiting_sum + self.states[j][9+self.actions[j]]
                  self.charging_sum = self.charging_sum + fem.get_charging_costs(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.states[j],self.Map.distance_matrix,k=3)[self.actions[j]]
              else:
                  self.actual_agent_2 = self.actual_agent_2 + fem.get_actual_distance(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,self.actions[j],k=3)
          
          ret_cost = [0]*self.charging_requests
          total = 0 
          for j in range(self.charging_requests):
              if self.data.iloc[j,6]==0:
                  ret_cost[j]=fem.get_actual_costs(self.ev_requests.iloc[j,:].astype(int).tolist(),self.charging_stations,self.states[j],self.time_mat,self.Map.distance_matrix,self.charging_stations[self.actions[j]],self.actions[j],k=3)-fem.get_travel_costs(self.ev_requests.iloc[j,:].astype(int).tolist(),self.charging_stations,self.states[j],self.Map.distance_matrix,action,k=3)[self.actions[j]]
                  self.score_0 += (-ret_cost[j])
              else:
                  ret_cost[j] = fem.get_actual_distance(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,self.actions[j],k=3)-fem.get_distance_cost(self.data.iloc[j,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,self.actions[j],k=3)
                  self.score_1 += (-ret_cost[j])
              total = total + ret_cost[j]
          self.reward = -total 
        else:
          return_reward = 0
          if self.data.iloc[self.i-1,6]==0:
              return_reward = fem.get_travel_costs(self.data.iloc[self.i-1,:].astype(int).tolist(),self.charging_stations,self.observation,self.Map.distance_matrix,action,k=3)[action]
              self.score_0 += (-return_reward)
              self.eval_dist_0 += fem.get_distance_cost(self.data.iloc[self.i-1,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,action,k=3)
              self.eval_cost_0 += return_reward
          else:
              return_reward = fem.get_distance_cost(self.data.iloc[self.i-1,:].astype(int).tolist(),self.charging_stations,self.Map.distance_matrix,action,k=3)
              self.score_1 += (-return_reward)
              self.eval_dist_1 += return_reward
              self.eval_cost_1 += fem.get_travel_costs(self.data.iloc[self.i-1,:].astype(int).tolist(),self.charging_stations,self.observation,self.Map.distance_matrix,action,k=3)[action]
          self.reward = -return_reward
          
        
        self.score += self.reward
        self.info = {}
        self.truncated = False
        self.observation = self.next_state
        
        if self.terminated is True:
            '''
            print("The cumm travel costs for cost pref cars in an episode are: ", self.eval_cost_0)
            print("The cumm travel costs for dist pref cars in an episode are: ", self.eval_cost_1)
            print("The cumm dist costs for dist pref cars in an episode are: ", self.eval_dist_1)
            print("The cumm dist costs for cost pref cars in an episode are: ", self.eval_dist_0)
            '''
            self.total_actual_travel_costs.append(self.summ)
            self.total_episodic_charging_times.append(self.charging_sum)
            self.total_episodic_waiting_times.append(self.waiting_sum)
            self.total_episodic_times.append(self.time_sum)
            self.total_episodic_distances.append(self.dist_sum)
            self.scores.append(self.score)   
            self.scores_0.append(self.score_0)
            self.scores_1.append(self.score_1)
            self.agent_2.append(self.actual_agent_2)
            self.scores_window.append(self.score)      
            self.dist_window.append(self.dist_sum)
            self.time_window.append(self.time_sum)
            self.charging_window.append(self.charging_sum)
            self.waiting_window.append(self.waiting_sum)
            self.actual_window.append(self.summ)
            self.agent_2_dq.append(self.actual_agent_2)
            self.score_0_dq.append(self.score_0)
            self.score_1_dq.append(self.score_1)
            
            self.avg_episodic_rew.append(np.mean(self.scores_window))
            self.avg_distance.append(np.mean(self.dist_window))
            self.avg_time.append(np.mean(self.time_window))
            self.avg_charging_time.append(np.mean(self.charging_window))
            self.avg_waiting_time.append(np.mean(self.waiting_window))
            self.avg_actual_travel_cost.append(np.mean(self.actual_window))
            self.avg_agent_2.append(np.mean(self.agent_2_dq))
            self.avg_score_0.append(np.mean(self.score_0_dq))
            self.avg_score_1.append(np.mean(self.score_1_dq))
        return self.next_state, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None):
        self.data = generate_data(self.j,self.dist)
        self.ev = classes.EV(self.data.iloc[0,0],self.data.iloc[0,1],self.data.iloc[0,4],self.data.iloc[0,5], 0)
        self.observation = fem.preprocess_data(self.ev,self.data.iloc[0,:].astype(int).tolist(), self.charging_stations, self.Map, self.time_mat,self.charging_stations_info, 3)
        self.states = [self.observation]
        self.evs = [self.ev]
        self.actions = []
        self.info = {}
        self.i = 0
        self.time_mat = self.Map.get_time(39)
        self.evcs1 = classes.EVCS(2,0.16)
        self.evcs2 = classes.EVCS(2,0.16)
        self.evcs3 = classes.EVCS(2,0.16)
        self.charging_stations_info = [self.evcs1,self.evcs2,self.evcs3]
        self.counter = 0 
        self.summ = 0
        self.dist_sum = 0 
        self.time_sum = 0
        self.waiting_sum = 0
        self.charging_sum = 0
        self.score = 0
        self.score_0 = 0
        self.score_1 = 0
        self.eval_cost_0 = 0 
        self.eval_cost_1 = 0 
        self.eval_dist_0 = 0 
        self.eval_dist_1 = 0
        self.actual_agent_2 = 0 
        return self.observation, self.info
    