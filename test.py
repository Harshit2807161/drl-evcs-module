# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:51:16 2024

@author: Harshit
"""

import numpy as np
import matplotlib.pyplot as plt

def sample_normal_integers_and_plot(low, high, num_samples):
    """
    Sample integers from a normal distribution with mean at the midpoint of the range,
    and arrange them in ascending order. Plot the samples along with the probability density function (PDF)
    of the normal distribution.

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
    
    # Plot the histogram of sampled integers
    plt.hist(sorted_samples, bins=np.arange(low - 0.5, high + 1.5, 1), density=True, alpha=0.5, label='Sampled Integers')
    
    # Plot the probability density function (PDF) of the normal distribution
    x = np.linspace(low - 0.5, high + 0.5, 1000)
    y = 1 / ((high - low) / 6 * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * ((high - low) / 6)**2))
    plt.plot(x, y, color='red', label='Normal Distribution PDF')
    
    plt.xlabel('Integer Values')
    plt.ylabel('Probability Density')
    plt.title('Sampled Integers and Normal Distribution PDF')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return sorted_samples.tolist()

# Example usage:
low = 20
high = 100
num_samples = 120
sample_normal_integers_and_plot(low, high, num_samples)
