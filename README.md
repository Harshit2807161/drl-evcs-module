# Deep Reinforcement Learning for EV Charging Station Selection (DRL-EVCS-MODULE)

This repository contains the codebase for the research paper **"Deep Reinforcement Learning for Grid-Based Charging Station Selection"**. The project explores the application of Deep Reinforcement Learning (DRL) techniques to optimize the selection of electric vehicle (EV) charging stations, aiding in automated EV navigation with an emphasis on minimizing cost or distance based on driver preferences.

Keywords: *Charging navigation, Driver preferences, Deep reinforcement learning, Intelligent transport systems, Plug-in electric vehicles (PEVs), Proximal Policy Optimization.*

## Directory Structure

```
DRL-EVCS-MODULE/
├── checkenv.py         # Sanity check for the custom environment
├── classes.py          # Contains core classes for EV, EVCS, ITS, and MAP with relevant methods and attributes
├── DDQNtrain.py        # Script to run DDQN agent on the custom environment
├── DQNtrain.py         # Script to run DQN agent on the custom environment
├── fem.py              # Helper functions for state transitions, waiting times, and preprocessing
├── generate_uniform.py # Helper script to generate uniform car distributions in the map
├── main.py             # Environment initialization, action/state transitions, and plotting variables
├── PPOtrain.py         # Script to run PPO agent on the custom environment
├── README.md           # Project documentation
├── sample_normal.py    # Script for sampling car distributions using a normal distribution
└── test_agent.py       # Script to test trained RL agents
```

## Key Components

- **checkenv.py**: A sanity check script designed to validate the custom environment. Recommended to be run after any modification to the environment setup.
  
- **classes.py**: Contains the definitions for various classes, including:
  - `EV` (Electric Vehicle): Attributes like position, destination, state of charge (soc), and required_soc along with helper methods.
  - `EVCS` (Electric Vehicle Charging Station): Methods like `confirm_reservation`, `remove_ev`, `find_queue_number`, and `waiting_time`.
  - `MAP`: Stores grid information (like adjacency matrix and average velocities on each edge (road)), and handles grid-related computations like `get_time`, `insert_traffic_values`, and `insert_distance_values`.
  
- **DDQNtrain.py**: A training script for the **DDQN** (Double Deep Q-Network) agent. You can modify hyperparameters inline for different experimental setups.

- **DQNtrain.py**: A training script for the **DQN** (Deep Q-Network) agent.

- **main.py**: Defines the RL environment variables, state transitions (`step` and  `reset` function), and sets up variables for plotting results and performance metrics.

- **fem.py**: Contains a series of helper methods, including `djikstra_with_path`, `get_expected_waiting_times`, `get_arrival_times`, and `preprocess_data` for state preprocessing.

- **generate_uniform.py**: Script to generate car distributions uniformly on the map.

- **PPOtrain.py**: A training script for the **PPO** (Proximal Policy Optimization) agent.

- **sample_normal.py**: Pre-samples car distributions on the map using a normal distribution.

- **test_agent.py**: Used for testing trained agent to evaluate their performance on unseen data.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DRL-EVCS-MODULE.git
   ```

2. To validate the environment, run:
   ```bash
   python checkenv.py
   ```

## How to Run

- To train a **DQN agent**, use:
  ```bash
  python DQNtrain.py
  ```

- To train a **DDQN agent**, use:
  ```bash
  python DDQNtrain.py
  ```

- To train a **PPO agent**, use:
  ```bash
  python PPOtrain.py
  ```

- You can modify the hyperparameters for each model inline within the training scripts. Example parameters include learning rate, epsilon decay, discount factor, batch size, etc.

- To test an agent after training:
  ```bash
  python test_agent.py
  ```

## Contributions

This repository is actively maintained for the purpose of the paper review and is currently private. 
## License

This project is licensed under the MIT License.

---
