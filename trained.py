from main import EVRL
from stable_baselines3 import PPO,DQN

j = 80
dist = 'N'
env = EVRL(j+1,dist)
env.reset()
s =j*2000
model_path = f"models/DDQN/{dist}/{j}/1712522273/{s}.zip"
model = DQN.load(model_path, env = env)
episodes = 2

for ep in range(episodes):
    obs,_ = env.reset()
    done = False
    pref = obs[6]
    print(f"Agent {ep+1} information: \n")
    for i in range(2):
        action,_ = model.predict(obs)
        a = env.charging_stations[action]
        if obs[6]==0:
            print("When the preference for agent is travel costs: ")
        else:
            print("When the preference for agent is distance costs: ")
        obs, reward, done, trunc, info = env.step(action)
        
        
    print("\n")


