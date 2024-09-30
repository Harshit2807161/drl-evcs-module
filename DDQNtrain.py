from main import EVRL
import os
import time
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import csv

NO_OF_CARS = [80,100,120,140]
dist = 'N'
for j in NO_OF_CARS:
    model_dir = f"models/DDQN/{dist}/{j}/{int(time.time())}"
    log_dir = f"logs/DDQN/{dist}/{j}/{int(time.time())}"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    env = EVRL(j+1,dist)
    env.reset()
    
    # Define and Train the agent
    model = DQN("MlpPolicy", env, verbose=1, tau=0.4, learning_rate=8e-3, learning_starts=30, double_dqn=True, exploration_fraction=0.7,  batch_size=128, tensorboard_log=log_dir, device="cuda", buffer_size=100000) 
    no_episodes = 2000
    TIMESTEPS = (j)*no_episodes
    print("Number of timesteps: ",TIMESTEPS)
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,tb_log_name="DDQN",progress_bar="True")
    model.save(f"{model_dir}/{TIMESTEPS}")
    
    save_dir = f"results/DDQN/{dist}/{j} cars/"
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.plot(env.scores,color='blue')
    plt.plot(env.avg_episodic_rew,color='red')
    plt.ylabel(f'Cumulative Rewards- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm reward.png')
    plt.show()

    csv_file = save_dir + "Cumm reward.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.scores, env.avg_episodic_rew):
            writer.writerow([num1, num2])
    
    
    plt.plot(env.scores_0,color='blue')
    plt.plot(env.avg_score_0,color='red')
    plt.ylabel(f'Cumulative travel cost in rewards- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm reward - travel_costs.png')
    plt.show()
    
    csv_file = save_dir + "Cumm reward - travel_costs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.scores_0, env.avg_score_0):
            writer.writerow([num1, num2])
    
    plt.plot(env.total_episodic_distances,color='blue')
    plt.plot(env.avg_distance,color='red')
    plt.ylabel(f'Cumulative episodic distances- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm distance.png')
    plt.show()
    
    csv_file = save_dir + "Cumm distance.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.total_episodic_distances, env.avg_distance):
            writer.writerow([num1, num2])
    
    plt.plot(env.total_episodic_times,color='blue')
    plt.plot(env.avg_time,color='red')
    plt.ylabel(f'Cumulative episodic travel times- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm time.png')
    plt.show()
    
    csv_file = save_dir + "Cumm time.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.total_episodic_times, env.avg_time):
            writer.writerow([num1, num2])
    
    plt.plot(env.total_episodic_waiting_times,color='blue')
    plt.plot(env.avg_waiting_time,color='red')
    plt.ylabel(f'Cumulative episodic waiting costs- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm waiting costs.png')
    plt.show()
    
    csv_file = save_dir + "Cumm waiting costs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.total_episodic_waiting_times, env.avg_waiting_time):
            writer.writerow([num1, num2])
    
    plt.plot(env.total_episodic_charging_times,color='blue')
    plt.plot(env.avg_charging_time,color='red')
    plt.ylabel(f'Cumulative episodic charging costs- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm charging costs.png')
    plt.show()
    
    csv_file = save_dir + "Cumm charging costs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.total_episodic_charging_times, env.avg_charging_time):
            writer.writerow([num1, num2])
    
    plt.plot(env.total_actual_travel_costs,color='blue')
    plt.plot(env.avg_actual_travel_cost,color='red')
    plt.ylabel(f'Actual episodic travel costs- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm actual travel costs.png')
    plt.show()
    
    csv_file = save_dir + "Cumm actual travel costs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.total_actual_travel_costs, env.avg_actual_travel_cost):
            writer.writerow([num1, num2])
    
    plt.plot(env.scores_1,color='blue')
    plt.plot(env.avg_score_1,color='red')
    plt.ylabel(f'Cumulative distance cost in rewards- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm reward - distance_costs.png')
    plt.show()
    
    csv_file = save_dir + "Cumm reward - distance_costs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.scores_1, env.avg_score_1):
            writer.writerow([num1, num2])
    
    plt.plot(env.agent_2,color='blue')
    plt.plot(env.avg_agent_2,color='red')
    plt.ylabel(f'Actual episodic distance costs- {j} cars')
    plt.xlabel('Episode No#')
    plt.savefig(save_dir + 'Cumm actual distance costs.png')
    plt.show()
    
    csv_file = save_dir + "Cumm actual distance costs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cumm", "Average"])  # Write headings
        for num1, num2 in zip(env.agent_2, env.avg_agent_2):
            writer.writerow([num1, num2])
