import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os
import time
from collections import defaultdict
from main import EVRL
from generate_uniform import generate_data 

NO_OF_CARS_OPTIONS = [80, 100, 120, 140] # Number of EV requests per episode
DISTRIBUTION = 'U' # or 'U', as used in your EVRL and DQNtrain
NUM_EPISODES = 2000 # Total number_of_episodes for training

ALPHA = 0.1  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0  # Starting exploration rate
EPSILON_END = 0.01  # Minimum exploration rate
EPSILON_DECAY_EPISODES = NUM_EPISODES * 0.8

# --- Helper Function for Epsilon-Greedy Policy ---
def choose_action_epsilon_greedy(q_table, state_tuple, epsilon, action_space):
    """
    Chooses an action using an epsilon-greedy policy.
    Args:
        q_table (defaultdict): The Q-table.
        state_tuple (tuple): The current state (must be hashable).
        epsilon (float): The current exploration rate.
        action_space (gym.spaces.Discrete): The action space of the environment.
    Returns:
        int: The chosen action.
    """
    if random.uniform(0, 1) < epsilon:
        return action_space.sample()  # Explore
    else:
        # Exploit: Choose the action with the highest Q-value for the current state.
        # If state_tuple not in q_table, defaultdict will create it with zeros.
        return np.argmax(q_table[state_tuple])

# --- Q-learning Training Function ---
def train_q_learning(env, num_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay_episodes):
    """
    Trains an agent using the Q-learning algorithm.
    Args:
        env (EVRL): The EV routing environment.
        num_episodes (int): The number of episodes to train for.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon_start (float): Initial epsilon value.
        epsilon_end (float): Final epsilon value.
        epsilon_decay_episodes (float): Number of episodes over which to decay epsilon.
    Returns:
        defaultdict: The learned Q-table.
    """
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    current_epsilon = epsilon_start
    epsilon_decay_value = (epsilon_start - epsilon_end) / epsilon_decay_episodes if epsilon_decay_episodes > 0 else 0

    print(f"Starting Q-learning training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset() 
        state_tuple = tuple(state) 
        
        terminated = False
        truncated = False
        episode_reward_sum = 0

        while not terminated and not truncated:
            action = choose_action_epsilon_greedy(q_table, state_tuple, current_epsilon, env.action_space)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_tuple = tuple(next_state)
            
            episode_reward_sum += reward

            # Q-learning update rule
            old_q_value = q_table[state_tuple][action]
            next_max_q = np.max(q_table[next_state_tuple]) 
            
            new_q_value = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)
            q_table[state_tuple][action] = new_q_value
            
            state_tuple = next_state_tuple

        if current_epsilon > epsilon_end:
            current_epsilon -= epsilon_decay_value
            current_epsilon = max(epsilon_end, current_epsilon)

        if (episode + 1) % 100 == 0: # Print progress
            print(f"Q-learning: Episode {episode + 1}/{num_episodes} completed. Epsilon: {current_epsilon:.4f}")
            if env.scores:
                 print(f"  Last episode reward (from env): {env.scores[-1]:.2f}")


    print("Q-learning training finished.")
    return q_table

# --- SARSA Training Function ---
def train_sarsa(env, num_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay_episodes):
    """
    Trains an agent using the SARSA algorithm.
    Args:
        env (EVRL): The EV routing environment.
        num_episodes (int): The number of episodes to train for.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon_start (float): Initial epsilon value.
        epsilon_end (float): Final epsilon value.
        epsilon_decay_episodes (float): Number of episodes over which to decay epsilon.
    Returns:
        defaultdict: The learned Q-table.
    """
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    current_epsilon = epsilon_start
    epsilon_decay_value = (epsilon_start - epsilon_end) / epsilon_decay_episodes if epsilon_decay_episodes > 0 else 0
    
    print(f"Starting SARSA training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset() # env handles its internal metrics reset here
        state_tuple = tuple(state)
        
        action = choose_action_epsilon_greedy(q_table, state_tuple, current_epsilon, env.action_space)
        
        terminated = False
        truncated = False
        episode_reward_sum = 0

        while not terminated and not truncated:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_tuple = tuple(next_state)
            
            episode_reward_sum += reward

            next_action = choose_action_epsilon_greedy(q_table, next_state_tuple, current_epsilon, env.action_space)
            
            # SARSA update rule
            old_q_value = q_table[state_tuple][action]
            next_q_value_for_sarsa = q_table[next_state_tuple][next_action] # Q-value of next_state, next_action
            
            new_q_value = old_q_value + alpha * (reward + gamma * next_q_value_for_sarsa - old_q_value)
            q_table[state_tuple][action] = new_q_value
            
            state_tuple = next_state_tuple
            action = next_action
            
        # Decay epsilon
        if current_epsilon > epsilon_end:
            current_epsilon -= epsilon_decay_value
            current_epsilon = max(epsilon_end, current_epsilon)

        if (episode + 1) % 100 == 0: # Print progress
            print(f"SARSA: Episode {episode + 1}/{num_episodes} completed. Epsilon: {current_epsilon:.4f}")
            if env.scores:
                 print(f"  Last episode reward (from env): {env.scores[-1]:.2f}")

    print("SARSA training finished.")
    return q_table


# --- Expected SARSA Training Function (Similar to TD(0) control) ---
def train_expected_sarsa(env, num_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay_episodes):
    """
    Trains an agent using the Expected SARSA algorithm.
    Args:
        env (EVRL): The EV routing environment.
        num_episodes (int): The number of episodes to train for.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon_start (float): Initial epsilon value.
        epsilon_end (float): Final epsilon value.
        epsilon_decay_episodes (float): Number of episodes over which to decay epsilon.
    Returns:
        defaultdict: The learned Q-table.
    """
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    current_epsilon = epsilon_start
    epsilon_decay_value = (epsilon_start - epsilon_end) / epsilon_decay_episodes if epsilon_decay_episodes > 0 else 0

    print(f"Starting Expected SARSA training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset()
        state_tuple = tuple(state)

        terminated = False
        truncated = False
        episode_reward_sum = 0

        while not terminated and not truncated:
            action = choose_action_epsilon_greedy(q_table, state_tuple, current_epsilon, env.action_space)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_tuple = tuple(next_state)

            episode_reward_sum += reward

            # Calculate the Expected Q-value for the next state under the current epsilon-greedy policy
            expected_q_next_state = 0
            if not (terminated or truncated): # Only calculate if next_state is not terminal
                num_actions = env.action_space.n
                q_values_next_state = q_table[next_state_tuple]
                greedy_action = np.argmax(q_values_next_state)

                # Probability of choosing the greedy action
                prob_greedy = 1 - current_epsilon + current_epsilon / num_actions
                expected_q_next_state += prob_greedy * q_values_next_state[greedy_action]

                # Probability of choosing any non-greedy action
                prob_nongreedy = current_epsilon / num_actions
                for a_prime in range(num_actions):
                    if a_prime != greedy_action:
                        expected_q_next_state += prob_nongreedy * q_values_next_state[a_prime]

            # Expected SARSA update rule
            old_q_value = q_table[state_tuple][action]
            new_q_value = old_q_value + alpha * (reward + gamma * expected_q_next_state - old_q_value)
            q_table[state_tuple][action] = new_q_value

            state_tuple = next_state_tuple

        # Decay epsilon
        if current_epsilon > epsilon_end:
            current_epsilon -= epsilon_decay_value
            current_epsilon = max(epsilon_end, current_epsilon)

        if (episode + 1) % 100 == 0: # Print progress
            print(f"Expected SARSA: Episode {episode + 1}/{num_episodes} completed. Epsilon: {current_epsilon:.4f}")
            if env.scores:
                 print(f"  Last episode reward (from env): {env.scores[-1]:.2f}")

    print("Expected SARSA training finished.")
    return q_table


# --- Plotting and CSV Saving Function ---
def save_results(env, base_save_dir, num_cars_str):
    """
    Saves plots and CSV files of the training metrics.
    Args:
        env (EVRL): The environment instance containing the metrics.
        base_save_dir (str): The base directory to save results.
        num_cars_str (str): String representing the number of cars for file naming.
    """
    save_dir = os.path.join(base_save_dir, f"{num_cars_str} cars")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created saving directory: {save_dir}") # Added confirmation print

    metrics_to_plot = [
        ("Cumm reward", env.scores, env.avg_episodic_rew),
        ("Cumm reward - travel_costs", env.scores_0, env.avg_score_0),
        ("Cumm reward - distance_costs", env.scores_1, env.avg_score_1),
        ("Cumm distance", env.total_episodic_distances, env.avg_distance),
        ("Cumm time", env.total_episodic_times, env.avg_time),
        ("Cumm waiting costs", env.total_episodic_waiting_times, env.avg_waiting_time),
        ("Cumm charging costs", env.total_episodic_charging_times, env.avg_charging_time),
        ("Cumm actual travel costs", env.total_actual_travel_costs, env.avg_actual_travel_cost),
        ("Cumm actual distance costs", env.agent_2, env.avg_agent_2),
    ]

    print(f"Saving results for {num_cars_str} cars to {save_dir}...") # Added start message

    for title_prefix, data_series, avg_data_series in metrics_to_plot:
        # Ensure data_series is a list or numpy array, not None
        if not data_series:
             print(f"Warning: Data series '{title_prefix}' is empty or None for {num_cars_str}. Skipping plot and CSV.")
             continue # Skip saving for this metric if data is empty

        plt.figure(figsize=(10, 6))
        plt.plot(data_series, color='blue', label='Per Episode')
        
        # Check avg_data_series as well if needed and not empty/None
        if avg_data_series and len(avg_data_series) > 0:
            # Ensure avg_data_series is also iterable and not empty if it exists
            try:
                 plt.plot(avg_data_series, color='red', label='Moving Average (Windowed)')
            except Exception as e:
                 print(f"Warning: Could not plot moving average for '{title_prefix}': {e}")
                 # Continue with saving the per-episode data if possible

        plt.ylabel(f'{title_prefix} - {num_cars_str} cars')
        plt.xlabel('Episode No#')
        plt.title(f'{title_prefix} vs. Episode - {num_cars_str} cars')
        plt.legend()
        plt.grid(True)

        # --- Save the plot ---
        plot_filename = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}.png")
        try:
            plt.savefig(plot_filename)
            # plt.show() # Comment out if running many experiments
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        finally:
             plt.close() # Always close the figure

        # --- Save the CSV ---
        csv_filename = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}.csv")
        try:
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["EpisodeValue", "MovingAverage"])
                # Use the actual length of data_series for zipping
                avg_data_to_zip = avg_data_series if avg_data_series and len(avg_data_series) == len(data_series) else [None] * len(data_series)
                
                for val, avg_val in zip(data_series, avg_data_to_zip):
                    writer.writerow([val, avg_val])
            print(f"Saved CSV: {csv_filename}")
        except Exception as e:
             print(f"Error saving CSV {csv_filename}: {e}")


    print(f"Finished saving results for {num_cars_str} cars.") # Added end message
    
# --- Main Execution ---
if __name__ == "__main__":
    # --- CHOOSE ALGORITHM ---
    ALGORITHM = "SARSA"  # Options: "Q_LEARNING" or "SARSA"

    for num_cars_setting in NO_OF_CARS_OPTIONS:        
        env_param_j = num_cars_setting # This `j` is passed to EVRL's __init__
        
        print(f"\n--- Training for {num_cars_setting} EV requests per episode ---")
        
        # Create directories for models (Q-tables) and logs/results
        timestr = time.strftime("%Y%m%d-%H%M%S")
        base_results_dir = f"results/{ALGORITHM}/{DISTRIBUTION}/{num_cars_setting}_cars_env_param_{env_param_j}/{timestr}"
        
        if not os.path.exists(base_results_dir):
            os.makedirs(base_results_dir)
            print(f"Created results directory: {base_results_dir}")

        # Initialize environment
        env = EVRL(j=env_param_j, dist=DISTRIBUTION)
        
# ... inside the main loop, after the training call ...

        if ALGORITHM == "Q_LEARNING":
            q_table = train_q_learning(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY_EPISODES)
        elif ALGORITHM == "SARSA":
            q_table = train_sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY_EPISODES)
            print(f"Debug: Checking metrics after {ALGORITHM} training for {num_cars_setting} cars...")
            print(f"Debug: Length of env.scores: {len(env.scores)}")
            print(f"Debug: Length of env.total_episodic_distances: {len(env.total_episodic_distances)}")
            if not env.scores or not env.total_episodic_distances:
                print(f"Debug: Warning: Metric lists appear empty or incomplete for {ALGORITHM} {DISTRIBUTION} {num_cars_setting} cars.")
                print("Debug: Saving results might fail.")
        elif ALGORITHM == "EXPECTED_SARSA":
            q_table = train_expected_sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY_EPISODES)
        else:
            raise ValueError("Invalid ALGORITHM specified. Choose 'Q_LEARNING' or 'SARSA' or 'EXPECTED_SARSA'.")

        # Save plots and CSVs
        save_results(env, base_results_dir, f"{num_cars_setting}_EVs")

# ... rest of the loop ...
        print(f"--- Finished training and saving results for {num_cars_setting} EV requests ---")

    print("\nAll training runs completed.")

