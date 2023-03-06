import pickle
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
    print(champion.head())
    champion.drop("crit", axis=1, inplace=True)
    champion.drop("critperlevel", axis=1, inplace=True)
    print(champion.head())

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
        if stat in ['Fighter', 'Mage', 'Marksman', 'Support', 'Tank', 'Assassin']:
            stats.append(1 if stat in champion[champion['id'] == champ]['tags'].values[0] else 0)
        else:
            stats.append(champion[champion['id'] == champ][stat].values[0])
    return stats

def getStat_red_blue(role, stat):
    """Cette fonction permet de récupérer les statistiques des champions qui ont joué le rôle donné pour les deux équipes

    : param role: le rôle du champion
    : param stat: la statistique à récupérer

    : return stats: deux listes contenant la statistique pour chaque champion
    """
    #on veut renvoyer la différence entre les stats des deux équipes pour un role donné
    print(role, stat)
    bleu = getStat('blue', role, stat)
    rouge = getStat('red', role, stat)
    stats = [bleu, rouge]
    return stats

def getStat_difference(role, stat):
    """Cette fonction permet de récupérer les statistiques du champion qui a joué le rôle donné pour l'équipe donnée

    : param role: le rôle du champion
    : param stat: la statistique à récupérer

    : return stats: une liste contenant la statistique pour chaque champion
    """
    #on veut renvoyer la différence entre les stats des deux équipes pour un role donné
    print(role, stat)
    bleu = getStat('blue', role, stat)
    rouge = getStat('red', role, stat)
    stats = [bleu[i] / ((rouge[i] if rouge[i]!=0 else 1)) for i in range(len(bleu))]
    return [stats]

def train_test_split(func: callable, *args, test_size: float = 0.2):
    """Cette fonction permet de diviser les données en un ensemble d'entraînement et un ensemble de test

    : param  func: la fonction qui permet de récupérer les données
    : param  args: les arguments de func
    : param  test_size: la taille de l'ensemble de test

    : return X_train: les données d'entraînement
    : return X_test: les données de test
    : return y_train: les labels d'entraînement
    : return y_test: les labels de test
    """
    X = [list(i) for i in zip(*sum([sum([func(arg[0], ar) for ar in arg[1]], []) for arg in args], []))] # Si ça marche, pas touche
    y = matches['result']
    n = len(X)
    cut = int((1-test_size)*n)
    X_train = X[:cut]
    X_test = X[cut:]
    y_train = y[:cut]
    y_test = y[cut:]
    return X_train, X_test, y_train, y_test

def getAccuracy(clf: tree.DecisionTreeClassifier, X_test: list, y_test: list):
    """Cette fonction permet de calculer la précision d'un classifieur

    : param  clf: le classifieur
    : param  X_test: les données de test
    : param  y_test: les labels de test

    : return accuracy: la précision du classifieur
    """
    accuracy = 0
    for i in range(len(X_test)):
        if clf.predict([X_test[i]]) == y_test.values[i]:
            accuracy += 1
    return accuracy/len(X_test)

def train(X_train: list, y_train: list, min_samples_split: int=2, max_depth: int=None):
    """Cette fonction permet d'entraîner un classifieur

    : param  X_train: les données d'entraînement
    : param  y_train: les labels d'entraînement
    : param  min_samples_split: le nombre minimum d'échantillons requis pour diviser un noeud
    : param  max_depth: la profondeur maximale de l'arbre

    : return clf: le classifieur
    """
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    return clf

def bestParamsplot(X_train: list, X_test: list, y_train: list, y_test: list, min_samples_split: list=[1, 2, 5, 10, 20, 50, 100, 1000], max_depth: list=[None, 2, 5, 10, 20, 50, 100]):
    """Cette fonction permet de trouver les meilleurs paramètres pour un classifieur et de tracer la précision en fonction de ces paramètres

    : param  X_train: les données d'entraînement
    : param  X_test: les données de test
    : param  y_train: les labels d'entraînement
    : param  y_test: les labels de test
    : param  min_samples_split: la liste des valeurs de min_samples_split à tester
    : param  max_depth: la liste des valeurs de max_depth à tester

    : return bestParams: les meilleurs paramètres
    """
    bestParams = {'min_samples_split': 0, 'max_depth': 0, 'accuracy': 0}
    accuracy = []
    for i in min_samples_split:
        print(bestParams)
        accuracy.append([])
        for j in max_depth:
            clf = train(X_train, y_train, i, j)
            accuracy[-1].append(getAccuracy(clf, X_test, y_test))
            # print('min_samples_split =', i, 'max_depth =', j, 'accuracy =', accuracy[-1][-1])
            if accuracy[-1][-1] > bestParams['accuracy']:
                bestParams['min_samples_split'] = i
                bestParams['max_depth'] = j
                bestParams['accuracy'] = accuracy[-1][-1]
    plt.figure("Accuracy")
    sns.heatmap(accuracy, xticklabels=max_depth, yticklabels=min_samples_split, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Accuracy")
    plt.xlabel('max_depth')
    plt.ylabel('min_samples_split')
    plt.show()
    return bestParams

# # Exemple d'utilisation des fonctions
# X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp',)), test_size=0.2)
# clf = train(X_train, y_train, 100, 5)
# print(getAccuracy(clf, X_test, y_test)) # 0.5125786163522013, pas super mais mieux que rien

# # bestParams automatise la recherche des meilleurs paramètres (min_samples_split et max_depth)
# params = bestParamsplot(getStat_red_blue, ('top', ('hp',)), test_size=0.2)
# print(params)

# # bestParamsplot affiche aussi une heatmap des résultats
# params = bestParamsplot(getStat_red_blue, ('top', ('hp',)), test_size=0.2)
# print(params)

# # Les hps seul ne sont pas très efficaces, on peut essayer avec les hps et les mps
# X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp', 'mp')), test_size=0.2)
# clf = train(X_train, y_train, 100, 5)
# print(getAccuracy(clf, X_test, y_test)) # 0.5157232704402516, un peu mieux, il ne reste plus qu'à mettre plus de paramètres

# params = bestParamsplot(getStat_red_blue, ('top', ('hp', 'mp')), test_size=0.2)
# print(params)

# # On va essayer avec toutes les stats (ça va probablement être long (effectivement, ça a mis 02h 07min... les meilleurs paramètres sont min_samples_split = 2 max_depth = 5))
stats = ('attack', 'defense', 'magic', 'difficulty', 'Fighter', 'Tank', 'Mage', 'Assassin', 'Support', 'Marksman', 'hp', 'hpperlevel', 'mp', 'mpperlevel', 'movespeed', 'armor', 'armorperlevel', 'spellblock', 'spellblockperlevel', 'attackrange', 'hpregen', 'hpregenperlevel', 'mpregen', 'mpregenperlevel', 'attackdamage', 'attackdamageperlevel', 'attackspeedperlevel', 'attackspeed')
roles = ('top', 'jungle', 'mid', 'adc', 'support')
# # params = bestParamsplot(getStat_red_blue, *[(pos, stat) for pos in roles], test_size=0.2)
# # print(params)

# On va sauvegarder le meilleur arbre et le dataset pour ne pas les générer à chaque fois
X_train, X_test, y_train, y_test = train_test_split(getStat_difference, *[(pos, stats) for pos in roles], test_size=0.2)
print(X_test)
pickle.dump(X_train, open('full_X_train_difference.pkl', 'wb'))
pickle.dump(X_test, open('full_X_test_difference.pkl', 'wb'))
pickle.dump(y_train, open('full_y_train_difference.pkl', 'wb'))
pickle.dump(y_test, open('full_y_test_difference.pkl', 'wb'))

X_train = pickle.load(open('full_X_train_difference.pkl', 'rb'))
X_test = pickle.load(open('full_X_test_difference.pkl', 'rb'))
y_train = pickle.load(open('full_y_train_difference.pkl', 'rb'))
y_test = pickle.load(open('full_y_test_difference.pkl', 'rb'))

clf = train(X_train, y_train, 2, 3) # prends ~ 3min
pickle.dump(clf, open('full_tree_difference.pkl', 'wb'))# 2 3 <- mettre à jour si on change les paramètres

clf = pickle.load(open('full_tree.pkl', 'rb'))
tree.plot_tree(clf)
print("accuracy = ", getAccuracy(clf, X_test, y_test)) # 0.5293501048218029 avec 2 3
plt.show()

params = bestParamsplot(X_train, X_test, y_train, y_test, range(1, 50), range(1, 50))
print(params)

# print(matches['bluetop'])
# clf = tree.DecisionTreeClassifier(min_samples_split=100, max_depth=5)
# clf = clf.fit([[i,j] for i, j in zip(getStats(matches['bluetop'],'hp'), getStats(matches['redtop'],'hp'))], matches['result'])
# tree.plot_tree(clf)
# plt.show()
# entrainer un arbre a partir de getstat_difference 
# train_test_split(getStat_difference, ('top', ('attack',)), test_size=0.5)
# clf = tree.DecisionTreeClassifier(min_samples_split=100, max_depth=5)

