import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree

if True: # Lecture des données
    champion = pd.read_csv('champions.csv', index_col=None)
    # Les tags sont encore sous forme de chaîne de caractères, nous devons les convertir en listes
    champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
    matches = pd.read_csv('matches.csv', index_col=None)
    print( " I got the datas ! ")

# Helper functions
def getStat(color, role, stat):
    """Cette fonction permet de récupérer les statistiques du champion qui a joué le rôle donné pour l'équipe donnée
    
    : param color: la couleur de l'équipe
    : param role: le rôle du champion
    : param stat: la statistique à récupérer
    : return stats: une liste contenant la statistique pour chaque champion
    """
    stats = []
    for champ in matches[color + role]:
        stats.append(champion[champion['id'] == champ][stat].values[0])
    return stats

def train_test_split(func: callable, *args, test_size: float=0.2):
    """Cette fonction permet de diviser les données en un ensemble d'entraînement et un ensemble de test
    : param  func: la fonction qui permet de récupérer les données
    : param  test_size: la taille de l'ensemble de test
    : return X_train: les données d'entraînement
    : return X_test: les données de test
    : return y_train: les labels d'entraînement
    : return y_test: les labels de test
    """
    X = [list(i) for i in zip(*sum([[func('red', arg[0], ar) for ar in arg[1]] for arg in args] + [[func('blue', arg[0], ar) for ar in arg[1]] for arg in args], []))] # Si ça marche, pas touche
    y = matches['result']
    n = len(X)
    cut = int((1-test_size)*n)
    X_train = X[:cut]
    X_test = X[cut:]
    y_train = y[:cut]
    y_test = y[cut:]
    return X_train, X_test, y_train, y_test

def getAccuracy(clf: tree.DecisionTreeClassifier, X_test: list, y_test: list):
    # Cette fonction permet de calculer la précision d'un classifieur
    # Entrée:
    #   - clf: le classifieur
    #   - X_test: les données de test
    #   - y_test: les labels de test
    # Sortie:
    #   - accuracy: la précision du classifieur
    accuracy = 0
    for i in range(len(X_test)):
        if clf.predict([X_test[i]]) == y_test.values[i]:
            accuracy += 1
    return accuracy/len(X_test)

def train(X_train: list, y_train: list, min_samples_split: int=2, max_depth: int=None):
    # Cette fonction permet d'entraîner un classifieur
    # Entrée:
    #   - func: la fonction qui permet de récupérer les données
    #   - test_size: la taille de l'ensemble de test
    #   - min_samples_split: le nombre minimum d'exemples pour diviser un noeud
    #   - max_depth: la profondeur maximale de l'arbre
    # Sortie:
    #   - clf: le classifieur
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    return clf

def bestParams(func: callable, *args, test_size: float=0.2, min_samples_split: list=[2, 5, 10, 20, 50, 100], max_depth: list=[None, 2, 5, 10, 20, 50, 100]):
    # Cette fonction permet de trouver les meilleurs paramètres pour un classifieur
    # Entrée:
    #   - func: la fonction qui permet de récupérer les données
    #   - test_size: la taille de l'ensemble de test
    #   - min_samples_split: la liste des valeurs de min_samples_split à tester
    #   - max_depth: la liste des valeurs de max_depth à tester
    # Sortie:
    #   - bestParams: les meilleurs paramètres
    bestParams = {'min_samples_split': 0, 'max_depth': 0, 'accuracy': 0}
    for i in min_samples_split:
        for j in max_depth:
            X_train, X_test, y_train, y_test = train_test_split(func, *args, test_size=test_size)
            clf = train(X_train, y_train, i, j)
            accuracy = getAccuracy(clf, X_test, y_test)
            # print('min_samples_split =', i, 'max_depth =', j, 'accuracy =', accuracy)
            if accuracy > bestParams['accuracy']:
                bestParams['min_samples_split'] = i
                bestParams['max_depth'] = j
                bestParams['accuracy'] = accuracy
    return bestParams

def bestParamsplot(func: callable, *args, test_size: float=0.2, min_samples_split: list=[2, 5, 10, 20, 50, 100], max_depth: list=[None, 2, 5, 10, 20, 50, 100]):
    # Cette fonction permet de trouver les meilleurs paramètres pour un classifieur et d'afficher une heatmap
    # Entrée:
    #   - func: la fonction qui permet de récupérer les données
    #   - test_size: la taille de l'ensemble de test
    #   - min_samples_split: la liste des valeurs de min_samples_split à tester
    #   - max_depth: la liste des valeurs de max_depth à tester
    # Sortie:
    #   - bestParams: les meilleurs paramètres
    bestParams = {'min_samples_split': 0, 'max_depth': 0, 'accuracy': 0}
    accuracy = []
    for i in min_samples_split:
        accuracy.append([])
        for j in max_depth:
            X_train, X_test, y_train, y_test = train_test_split(func, *args, test_size=test_size)
            clf = train(X_train, y_train, i, j)
            accuracy[-1].append(getAccuracy(clf, X_test, y_test))
            # print('min_samples_split =', i, 'max_depth =', j, 'accuracy =', accuracy[-1][-1])
            if accuracy[-1][-1] > bestParams['accuracy']:
                bestParams['min_samples_split'] = i
                bestParams['max_depth'] = j
                bestParams['accuracy'] = accuracy[-1][-1]
    plt.figure("Accuracy")
    sns.heatmap(accuracy, xticklabels=max_depth, yticklabels=min_samples_split)
    plt.xlabel('max_depth')
    plt.ylabel('min_samples_split')
    plt.show()
    return bestParams

# Exemple d'utilisation des fonctions
X_train, X_test, y_train, y_test = train_test_split(getStat, ('top', ('hp',)), test_size=0.2)
clf = train(X_train, y_train, 100, 5)
print(getAccuracy(clf, X_test, y_test)) # 0.5125786163522013, pas super mais mieux que rien

# bestParams automatise la recherche des meilleurs paramètres (min_samples_split et max_depth)
params = bestParams(getStat, ('top', ('hp',)), test_size=0.2)
print(params)

# bestParamsplot affiche aussi une heatmap des résultats
params = bestParamsplot(getStat, ('top', ('hp',)), test_size=0.2)
print(params)

# # Les hps seul ne sont pas très efficaces, on peut essayer avec les hps et les mps
# X_train, X_test, y_train, y_test = train_test_split(getStat, ('top', ('hp', 'mp')), test_size=0.2)
# print(X_test)

# params = bestParamsplot(getStat, ('top', ('hp', 'mp')), test_size=0.2)
# print(params)

# print(matches['bluetop'])
# clf = tree.DecisionTreeClassifier(min_samples_split=100, max_depth=5)
# clf = clf.fit([[i,j] for i, j in zip(getStats(matches['bluetop'],'hp'), getStats(matches['redtop'],'hp'))], matches['result'])
# tree.plot_tree(clf)
# plt.show()
