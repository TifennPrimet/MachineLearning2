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
# Plot the mean and standard deviation of the numerical columns (there are a lot so we need the x axis legend to be sideways)
champion.plot(kind='box', title='Mean and standard deviation of the numerical columns') # Not very helpful
plt.xticks(rotation=45)
# Print the mean and standard deviation of the numerical columns
print(champion.describe())
# Let's try to look at how stats correlate with tags
# First, let's look at the tags
# Tags are stored as a list of strings, let's get a list of all the tags
tags = []
for tag in champion['tags']:
    tags.extend(tag)
# Now let's show the number of champions with each tag
plt.figure("Number of champions with each tag")
plt.hist(tags)
plt.xlabel('Tag')
plt.ylabel('Number of champions')
# Show the correlation between tags and roles
# Roles are bluetop, bluejungle, bluemid, blueadc, bluesupport, redtop, redjungle, redmid, redadc, redsupport and are columns in the matches data
# First, let's get a list of all the roles
roles = matches.columns[3:-1]
# Now let's get a list of all the unique tags
unique_tags = list(set(tags))
# Now let's make a dataframe with the number of champions with each tag for each role
tag_role = pd.DataFrame(index=unique_tags, columns=roles)
for role in roles:
    for tag in unique_tags:
        tag_role.loc[tag, role] = sum(matches[role].isin(champion[champion['tags'].apply(lambda x: tag in x)].index))
# Regroup the blue and red roles
tag_role['top'] = tag_role['bluetop'] + tag_role['redtop']
tag_role['jungle'] = tag_role['bluejungle'] + tag_role['redjungle']
tag_role['mid'] = tag_role['bluemid'] + tag_role['redmid']
tag_role['adc'] = tag_role['blueadc'] + tag_role['redadc']
tag_role['support'] = tag_role['bluesupport'] + tag_role['redsupport']
tag_role = tag_role.drop(['bluetop', 'bluejungle', 'bluemid', 'blueadc', 'bluesupport', 'redtop', 'redjungle', 'redmid', 'redadc', 'redsupport'], axis=1)
# flip the dataframe so that the roles are the index
tag_role = tag_role.transpose()
# Now let's plot the data
tag_role.plot(kind='bar', stacked=True, title='Number of champions with each tag for each role')
plt.xlabel('Role')
plt.ylabel('Number of champions')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Now let's look at the correlation between stats and roles
plt.show()