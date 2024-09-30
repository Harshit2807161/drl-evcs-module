from numpy import random as rd
import random
import pandas as pd
import numpy as np
def sample_uniform_integers(low, high, num_samples):
    """
    Sample integers from a uniform distribution within a specified range,
    and arrange them in ascending order.

    Args:
    - low: Lower boundary of the range (inclusive).
    - high: Upper boundary of the range (inclusive).
    - num_samples: Number of samples to be drawn.

    Returns:
    - sorted_samples: List of sampled integers arranged in ascending order.
    """
    # Sample integers from the uniform distribution
    samples = np.random.randint(low, high + 1, size=num_samples)
    # Sort the sampled integers in ascending order
    sorted_samples = np.sort(samples)
    return sorted_samples.tolist()

def sample_normal_integers(low, high, num_samples):
    """
    Sample integers from a normal distribution with mean at the midpoint of the range,
    and arrange them in ascending order.

    Args:
    - low: Lower boundary of the range (inclusive).
    - high: Upper boundary of the range (inclusive).
    - num_samples: Number of samples to be drawn.

    Returns:
    - sorted_samples: List of sampled integers arranged in ascending order.
    """
    # Calculate the mean of the range
    mean = (low + high) / 2
    
    # Sample integers from the normal distribution
    samples = np.random.normal(mean, (high - low) / 6, size=num_samples)  # Using (high - low) / 6 as standard deviation for 99.7% coverage
    
    # Round the sampled values to the nearest integers
    rounded_samples = np.round(samples).astype(int)
    
    # Clip the values to ensure they fall within the specified range
    rounded_samples = np.clip(rounded_samples, low, high)
    
    # Sort the sampled integers in ascending order
    sorted_samples = np.sort(rounded_samples)
    
    return sorted_samples.tolist()


def generate_data(x,dist):
  state_arr = []

  possible_postion = []

  for i in range(39):
      possible_postion.append(i)

  start_postion = 0
  end_position = 0
  battery = 0
  total_iterations = x
  iterations = 0 
  if dist=='U':
      deploy_times = sample_uniform_integers(20,100,x)
  if dist=='N':
      deploy_times = sample_normal_integers(20,100,x)
  time_gap = deploy_times[0]
  for i in deploy_times:
          new_state = []
          if iterations >= total_iterations:
              break
          start_postion = rd.choice(possible_postion)
          end_position = random.randint(0,38)

          while(end_position == start_postion):
              new_end_postion = rd.choice(possible_postion)
              end_position = new_end_postion
              
          battery = (rd.rand() * 0.2) + 0.2
          if len(state_arr) != 0:
              time_gap = i - state_arr[-1][2]
          if iterations <= total_iterations/2:
              l = 1
          else:
              l = 0
          new_state = [start_postion, end_position, i, time_gap, battery, 0.9, rd.randint(2)]
          state_arr.append(new_state)
          random.shuffle(possible_postion)
          iterations += 1 

  df = pd.DataFrame(state_arr, columns=['1', '2', '3', '4', '5', '6', '7'])
  df.at[0,'1'] = 13  
  df.at[0,'2'] = 17   
  df.at[0,'3'] = 20  
  df.at[0,'4'] = 20  
  df.at[0,'5'] = 0.322066  
  df.at[0,'6'] = 0.9  
  df.at[0,'7'] = 1
  df.iloc[1,:] = df.iloc[0,:]
  df.at[1,'4'] = 0
  if df.at[0,'7'] == 0:
      df.at[1,'7'] = 1
  else:
      df.at[1,'7'] = 0
  df.at[2,'4'] = df.at[2,'3'] - df.at[0,'3']  
  return df

generate_data(81, 'U')
