import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


# Read in the data
champion = pd.read_csv('champions.csv', index_col=0)
matches = pd.read_csv('matches.csv', index_col=0)
# Print the first 5 rows of the champions data
print(champion.head())
#plot the data
champion.plot() # Not very helpful
plt.figure()
plt.plot(champion['attack'], champion['magic'], 'o') # Not very helpful either
plt.xlabel('attack')
plt.ylabel('magic')
# Plot the mean and standard deviation of the numerical columns
champion.plot(kind='box') # Not very helpful
# Print the mean and standard deviation of the numerical columns
print(champion.describe())
plt.show()