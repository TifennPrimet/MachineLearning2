import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


# Read in the data
champion = pd.read_csv('champions.csv', index_col=0)
# Tags are still a string, so we need to make them a list
champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
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
# Let's try to look at how stats correlate with tags
# First, let's look at the tags
# Tags are stored as a list of strings, let's get a list of all the tags
tags = []
for tag in champion['tags']:
    tags.extend(tag)
# Now let's show the number of champions with each tag
plt.figure()
plt.hist(tags)
plt.xlabel('Tag')
plt.ylabel('Number of champions')
plt.show()