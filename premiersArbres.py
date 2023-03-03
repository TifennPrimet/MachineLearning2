import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree

if True: # Lecture des données
    # Lire les données
    champion = pd.read_csv('champions.csv', index_col=None)
    # Les tags sont encore sous forme de chaîne de caractères, nous devons les convertir en listes
    champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
    matches = pd.read_csv('matches.csv', index_col=None)

def getStats(champions, stat):
    # Cette fonction permet de récupérer les statistiques d'un champion
    # Entrée:
    #   - champions: la liste des champions
    #   - stat: la statistique à récupérer
    # Sortie:
    #   - stats: une liste contenant la statistique pour chaque champion
    stats = []
    for champ in champions:
        stats.append(champion[champion['id'] == champ][stat].values[0])
    return stats

print(matches['bluetop'])
clf = tree.DecisionTreeClassifier()
clf = clf.fit([[i,j] for i, j in zip(getStats(matches['bluetop'],'hp'), getStats(matches['redtop'],'hp'))], matches['result'])
