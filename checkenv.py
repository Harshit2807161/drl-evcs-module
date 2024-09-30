from stable_baselines3.common.env_checker import check_env
from main import EVRL


env = EVRL(100)
check_env(env)

# Just for sanity check on my custom environment. you can do the same if you make any changes to the environment wrapper.