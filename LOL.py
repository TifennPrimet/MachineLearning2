import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Lire les données
champion = pd.read_csv('champions.csv', index_col=None)
# Les tags sont encore sous forme de chaîne de caractères, nous devons les convertir en listes
champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
matches = pd.read_csv('matches.csv', index_col=None)
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
plt.xticks(rotation=90)
plt.tight_layout()
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
sns.histplot(tags, shrink=0.8, color='cornflowerblue')
plt.xlabel('Tag')
plt.ylabel('Nombre de champions')
plt.title('Nombre de champions pour chaque tag')
plt.tight_layout()
# Montrons la corrélation entre les tags et les rôles
# Les rôles sont "bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport", "redtop", "redjungle", "redmid", "redadc", "redsupport" et sont des colonnes dans les données de match
# Tout d'abord, obtenons une liste de tous les rôles
roles = matches.columns[4:-1]
print(roles)
# Maintenant, obtenons une liste de tous les tags uniques
unique_tags = list(set(tags))
# Maintenant, créons un dataframe avec le nombre de champions ayant chaque tag pour chaque rôle
tag_role = pd.DataFrame(index=unique_tags, columns=roles)
for role in roles:
    for tag in unique_tags:
        tag_role.loc[tag, role] = sum(matches[role].isin(champion[champion['tags'].apply(lambda x: tag in x)]['id']))
print(tag_role)

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
plt.tight_layout()

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
plt.tight_layout()

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
plt.tight_layout()

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
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# Faisons une heatmap pour voir quels tags sont le plus souvent ensemble
tag_tag = pd.DataFrame(index=unique_tags, columns=unique_tags, dtype=int)
for tag1 in unique_tags:
    for tag2 in unique_tags:
        tag_tag.loc[tag1, tag2] = sum(champion['tags'].apply(lambda x: tag1 in x and tag2 in x if tag1 != tag2 else tag1 in x and len(x) == 1))

# Tracer le résultat
plt.figure("Heatmap des tags")
sns.heatmap(tag_tag, annot=True, fmt='g', cmap='Blues')
plt.title('Heatmap des tags')
plt.xlabel('Tag 1')
plt.ylabel('Tag 2')
plt.tight_layout()

# Faisons une heatmap pour voir quels tags sont le plus souvent ensemble dans les matchs
tag_tag_m = pd.DataFrame(index=unique_tags, columns=unique_tags, dtype=int, data=0)
for tag1 in unique_tags:
    for tag2 in unique_tags:
        for champ in champion[champion['tags'].apply(lambda x: tag1 in x and tag2 in x if tag1 != tag2 else tag1 in x and len(x) == 1)]['id']:
            # Désolé pour la ligne suivante...
            tag_tag_m.loc[tag1, tag2] += sum(matches['bluetop'] == champ) + sum(matches['bluejungle'] == champ) + sum(matches['bluemid'] == champ) + sum(matches['blueadc'] == champ) + sum(matches['bluesupport'] == champ) + sum(matches['redtop'] == champ) + sum(matches['redjungle'] == champ) + sum(matches['redmid'] == champ) + sum(matches['redadc'] == champ) + sum(matches['redsupport'] == champ)
# Tracer le résultat
plt.figure("Heatmap des tags dans les matchs")
sns.heatmap(tag_tag_m, annot=True, fmt='g', cmap='Blues')
plt.title('Heatmap des tags dans les matchs')
plt.xlabel('Tag 1')
plt.ylabel('Tag 2')
plt.tight_layout()

# Faisons une heatmap pour voir quelles combinaisons de tags sont les plus populaires
# On va diviser tag_tag_m par tag_tag
tag_tag_ = tag_tag_m / tag_tag # Si quelqu'un a une meilleure idée pour tous ces noms, je suis preneur
# Tracer le résultat
plt.figure("Heatmap des tags dans les matchs normalisée")
sns.heatmap(tag_tag_, annot=True, fmt='.2f', cmap='Blues')
plt.title('Heatmap des tags dans les matchs normalisée')
plt.xlabel('Tag 1')
plt.ylabel('Tag 2')
plt.tight_layout()

# On va maintenant s'intéresser aux matchs : quels champions sont les plus populaires ? Quels sont les champions les plus efficaces ? Quels sont les champions les plus efficaces par rapport à leur popularité ?
# On va commencer par les champions les plus populaires

# On va créer un dataframe avec les champions et leur nombre d'apparition dans les matchs
champ_match = pd.DataFrame(index=champion['id'], columns=['popularite', 'victoires', 'defaites', 'taux victoire', 'taux defaite'])
for champ in champion['id']:
    champ_match.loc[champ] = sum(matches['bluetop'] == champ) + sum(matches['bluejungle'] == champ) + sum(matches['bluemid'] == champ) + sum(matches['blueadc'] == champ) + sum(matches['bluesupport'] == champ) + sum(matches['redtop'] == champ) + sum(matches['redjungle'] == champ) + sum(matches['redmid'] == champ) + sum(matches['redadc'] == champ) + sum(matches['redsupport'] == champ)

# On affiche les champions qui ne sont jamais apparus dans les matchs
champ_unused = champ_match[champ_match['popularite'] == 0]
print(champ_unused)
# Et on les enlève du dataframe
champ_match = champ_match[champ_match['popularite'] != 0]

blue_wins = matches[matches['result'] == 1]
red_wins = matches[matches['result'] == 0]

# On va maintenant créer un dataframe avec les champions et leur nombre de victoire
for champ in champion['id']:
    champ_match.loc[champ, 'victoires'] = sum(blue_wins['bluetop'] == champ) + sum(blue_wins['bluejungle'] == champ) + sum(blue_wins['bluemid'] == champ) + sum(blue_wins['blueadc'] == champ) + sum(blue_wins['bluesupport'] == champ) + sum(red_wins['redtop'] == champ) + sum(red_wins['redjungle'] == champ) + sum(red_wins['redmid'] == champ) + sum(red_wins['redadc'] == champ) + sum(red_wins['redsupport'] == champ)

# On va maintenant créer un dataframe avec les champions et leur nombre de défaite
for champ in champion['id']:
    champ_match.loc[champ, 'defaites'] = sum(red_wins['bluetop'] == champ) + sum(red_wins['bluejungle'] == champ) + sum(red_wins['bluemid'] == champ) + sum(red_wins['blueadc'] == champ) + sum(red_wins['bluesupport'] == champ) + sum(blue_wins['redtop'] == champ) + sum(blue_wins['redjungle'] == champ) + sum(blue_wins['redmid'] == champ) + sum(blue_wins['redadc'] == champ) + sum(blue_wins['redsupport'] == champ)

# On va maintenant créer un dataframe avec les champions et leur nombre de victoire par rapport à leur nombre d'apparition
for champ in champion['id']:
    if not champ in champ_unused.index:
        champ_match.loc[champ, 'taux victoire'] = champ_match.loc[champ, 'victoires'] / champ_match.loc[champ, 'popularite']
print(champ_match.sort_values(by='taux victoire', ascending=False).head(10))

# On va maintenant créer un dataframe avec les champions et leur nombre de défaite par rapport à leur nombre d'apparition
for champ in champion['id']:
    if not champ in champ_unused.index:
        champ_match.loc[champ, 'taux defaite'] = champ_match.loc[champ, 'defaites'] / champ_match.loc[champ, 'popularite']
print(champ_match.sort_values(by='taux defaite', ascending=False).head(10))

# On affiche tout ça
plt.figure("Champions les plus populaires")
champ_match.sort_values(by='popularite', ascending=False)["popularite"].head(10).plot(kind='bar')
plt.title('Champions les plus populaires')
plt.xlabel('Champion')
plt.ylabel('Nombre d\'apparition')
plt.tight_layout()

plt.figure("Champions les moins populaires")
champ_match.sort_values(by='taux victoire', ascending=False)["taux victoire"].head(10).plot(kind='bar')
plt.title('Champions les plus efficaces')
plt.xlabel('Champion')
plt.ylabel('Taux de victoire')
plt.tight_layout()

plt.figure("Champions les moins efficaces")
champ_match.sort_values(by='taux defaite', ascending=False)["taux defaite"].head(10).plot(kind='bar')
plt.title('Champions les moins efficaces')
plt.xlabel('Champion')
plt.ylabel('Taux de défaite')
plt.tight_layout()


plt.show()
