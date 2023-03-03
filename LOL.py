import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lire les données
champion = pd.read_csv('champions.csv', index_col=0)
# Les tags sont encore sous forme de chaîne de caractères, nous devons les convertir en listes
champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
matches = pd.read_csv('matches.csv', index_col=0)
# Afficher les 5 premières lignes des données sur les champions
print(champion.head())

# Tracer les données
champion.plot() # Pas très utile
plt.figure()
plt.plot(champion['attack'], champion['magic'], 'o') # Pas très utile non plus
plt.xlabel('attaque')
plt.ylabel('magie')

# Tracer la moyenne et l'écart type des colonnes numériques (il y en a beaucoup, donc nous devons mettre la légende de l'axe des x de côté)
champion.plot(kind='box', title='Moyenne et écart type des colonnes numériques') # Pas très utile
plt.xticks(rotation=45)

# Afficher la moyenne et l'écart type des colonnes numériques
print(champion.describe())

# Essayons de voir comment les statistiques sont corrélées aux tags
# Tout d'abord, regardons les tags
# Les tags sont stockés sous forme de liste de chaînes de caractères, obtenons une liste de tous les tags
tags = []
for tag in champion['tags']:
    tags.extend(tag)
# Montrons maintenant le nombre de champions avec chaque tag
plt.figure("Nombre de champions avec chaque tag")
plt.hist(tags)
plt.xlabel('Tag')
plt.ylabel('Nombre de champions')

# Montrons la corrélation entre les tags et les rôles
# Les rôles sont "bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport", "redtop", "redjungle", "redmid", "redadc", "redsupport" et sont des colonnes dans les données de match
# Tout d'abord, obtenons une liste de tous les rôles
roles = matches.columns[3:-1]
# Maintenant, obtenons une liste de tous les tags uniques
unique_tags = list(set(tags))
# Maintenant, créons un dataframe avec le nombre de champions ayant chaque tag pour chaque rôle
tag_role = pd.DataFrame(index=unique_tags, columns=roles)
for role in roles:
    for tag in unique_tags:
        tag_role.loc[tag, role] = sum(matches[role].isin(champion[champion['tags'].apply(lambda x: tag in x)].index))
# Regroupons les rôles bleus et rouges
tag_role['top'] = tag_role['bluetop'] + tag_role['redtop']
tag_role['jungle'] = tag_role['bluejungle'] + tag_role['redjungle']
tag_role['mid'] = tag_role['bluemid'] + tag_role['redmid']
tag_role['adc'] = tag_role['blueadc'] + tag_role['redadc']
tag_role['support'] = tag_role['bluesupport'] + tag_role['redsupport']
tag_role = tag_role.drop(['bluetop', 'bluejungle', 'bluemid', 'blueadc', 'bluesupport', 'redtop', 'redjungle', 'redmid', 'redadc', 'redsupport'], axis=1)
# Renverser le dataframe pour que les rôles soient les lignes et les tags les colonnes
tag_role = tag_role.transpose()
# Tracer le résultat
tag_role.plot(kind='bar', stacked=True, title='Nombre de champions avec chaque tag pour chaque rôle')
plt.xlabel('Rôle')
plt.ylabel('Nombre de champions')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Montrons maintenant la corrélation entre les tags et les statistiques
# Les stats sont attack, defense, magic, difficulty, hp, hpperlevel, mp, mpperlevel, movespeed, armor, armorperlevel, spellblock, spellblockperlevel, attackrange, hpregen, hpregenperlevel, mpregen, mpregenperlevel, crit, critperlevel, attackdamage, attackdamageperlevel, attackspeedperlevel, attackspeed et sont des colonnes numériques dans les données des champions
# On a déjà la liste des tags uniques
# Créons un dataframe avec la moyenne des statistiques pour chaque tag
tag_stat = pd.DataFrame(index=unique_tags, columns=champion.columns[2:]) # Contient aussi la colonne tags, on va devoir l'enlever
tag_stat = tag_stat.drop('tags', axis=1)
for tag in unique_tags:
    tag_stat.loc[tag] = champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['mean']
# Renverser le dataframe pour que les statistiques soient les lignes et les tags les colonnes
tag_stat = tag_stat.transpose()
# Tracer le résultat
tag_stat.plot(kind='bar', title='Moyenne des statistiques pour chaque tag') # Pas très utile, les moyennes dépendent de la stats mais pas vraiment du tag
plt.xlabel('Tag')
plt.ylabel('Moyenne')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Créons un dataframe avec l'écart type des statistiques pour chaque tag
tag_stat = pd.DataFrame(index=unique_tags, columns=champion.columns[2:]) # Contient aussi la colonne tags, on va devoir l'enlever
tag_stat = tag_stat.drop('tags', axis=1)
for tag in unique_tags:
    tag_stat.loc[tag] = champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['std']
# Renverser le dataframe pour que les statistiques soient les lignes et les tags les colonnes
tag_stat = tag_stat.transpose()
# Tracer le résultat
tag_stat.plot(kind='bar', title='Ecart type des statistiques pour chaque tag') # Pas très utile, les écart types dépendent trop de la moyenne
plt.xlabel('Tag')
plt.ylabel('Ecart type')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Créons un dataframe avec l'écart type relatif des statistiques pour chaque tag
tag_stat = pd.DataFrame(index=unique_tags, columns=champion.columns[2:]) # Contient aussi la colonne tags, on va devoir l'enlever
tag_stat = tag_stat.drop('tags', axis=1)
for tag in unique_tags:
    tag_stat.loc[tag] = champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['std'] / champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['mean']
# Renverser le dataframe pour que les statistiques soient les lignes et les tags les colonnes
tag_stat = tag_stat.transpose()
# Tracer le résultat
tag_stat.plot(kind='bar', title='Ecart type relatif des statistiques pour chaque tag') # Plus utile
plt.xlabel('Tag')
plt.ylabel('Ecart type relatif')
plt.show()
