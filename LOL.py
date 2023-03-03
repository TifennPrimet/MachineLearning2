import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


# Read in the data
champion = pd.read_csv('champions.csv', index_col=0)
matches = pd.read_csv('matches.csv', index_col=0)
# Print the first 5 rows of the champions data
print(champion.head())
#plot the data
champion.plot()
plt.figure()
plt.plot(champion['attack'], champion['magic'], 'o')
plt.xlabel('attack')
plt.ylabel('magic')
plt.show()