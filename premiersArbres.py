import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree, svm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

if 1==1: # Lecture des données
    champion = pd.read_csv('champions.csv', index_col=None)
    # Les tags sont encore sous forme de chaîne de caractères, nous devons les convertir en listes
    champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
    matches = pd.read_csv('matches.csv', index_col=None)

    print(champion.head())
    champion.drop("crit", axis=1, inplace=True)
    champion.drop("critperlevel", axis=1, inplace=True)
    print(champion.head())
    print( " I got the datas ! ")

if True:
    stats = pd.read_csv('full_stats.csv', index_col=None)

# Helper functions
def getStat_labonne(color, role, stat, new_datas):
    """Cette fonction permet de récupérer les statistiques du champion qui a joué le rôle donné pour l'équipe donnée
    
    : param color: la couleur de l'équipe
    : param role: le rôle du champion
    : param stat: la statistique à récupérer

    : return stats: une liste contenant la statistique pour chaque champion
    """
    stats = []
    for champ in new_datas[color + role]:
        if stat in ['Fighter', 'Mage', 'Marksman', 'Support', 'Tank', 'Assassin']:
            # print(champion[champion['id'] == champ]['tags'].values[0])
            
            stats.append(1 if f"'{stat}'" in champion[champion['id'] == champ]['tags'].values[0] else 0)
        else:
            stats.append(champion[champion['id'] == champ][stat].values[0])
    return stats


def getStat(color, role, stat):
    """Cette fonction permet de récupérer les statistiques du champion qui a joué le rôle donné pour l'équipe donnée

    : param color: la couleur de l'équipe
    : param role: le rôle du champion
    : param stat: la statistique à récupérer

    : return stats: une liste contenant la statistique pour chaque champion
    """
    return [color + role + stat]

def getStat_red_blue(role, stat):
    """Cette fonction permet de récupérer les statistiques des champions qui ont joué le rôle donné pour les deux équipes

    : param role: le rôle du champion
    : param stat: la statistique à récupérer

    : return stats: deux listes contenant la statistique pour chaque champion
    """
    return ['blue' + role + stat, 'red' + role + stat]

def getStat_difference(role, stat):
    """Cette fonction permet de récupérer les statistiques du champion qui a joué le rôle donné pour l'équipe donnée

    : param role: le rôle du champion
    : param stat: la statistique à récupérer

    : return stats: une liste contenant la statistique pour chaque champion
    """
    #on veut renvoyer la différence entre les stats des deux équipes pour un role donné
    print(role, stat)
    # ajouter une colonne stat'_diff' dans stats
    stats[stat + '_diff'] = stats['blue' + role + stat] - stats['red' + role + stat]
    return [stat + '_diff']

def getStat_rapport(role, stat):
    """Cette fonction permet de récupérer les statistiques du champion qui a joué le rôle donné pour l'équipe donnée

    : param role: le rôle du champion
    : param stat: la statistique à récupérer

    : return stats: une liste contenant la statistique pour chaque champion
    """
    #on veut renvoyer le rapport entre les stats des deux équipes pour un role donné
    print(role, stat)
    # ajouter une colonne stat'_rapport' dans stats
    stats[stat + '_rapport'] = stats['blue' + role + stat] / stats['red' + role + stat]
    # retirer les NaN
    stats[stat + '_rapport'] = stats[stat + '_rapport'].fillna(0)
    # retirer les inf
    stats[stat + '_rapport'] = stats[stat + '_rapport'].replace(np.inf, 0)
    return [stat + '_rapport']

def prepare_donnee(func: callable, *args):
    """Cette fonction permet de préparer les données pour l'entraînement

    : param  func: la fonction qui permet de récupérer les données
    : param  args: les arguments de func

    : return X: les données
    : return y: les labels
    """
    colonnes = []
    for arg in args:
        for ar in arg[1]:
            colonnes += func(arg[0], ar)
    X = stats[[col for col in colonnes]]
    y = stats['result']
    return X, y

def cross_validation(X, y, k, critere = 'gini'):
    """Cette fonction permet de faire une cross validation

    : param  X: les données
    : param  y: les labels
    : param  k: le nombre de folds

    : return scores: les scores de la cross validation
    """
    # On fait une copie des données pour ne pas modifier les données d'origine
    X = X.copy()
    y = y.copy()
    # On ajoute les résultats aux données pour tout spliter d'un coup
    X.insert(X.shape[1],'resultat',y)
    # On split les données en k folds
    X_split = np.array_split(X,k)
    scores = [] # Initialisation des scores de précision
    for i in range(k):
        # On récupère les données à prédire et celles à entraîner
        X_topred = X_split[i].copy()
        y_topred = X_topred['resultat']
        X_totrain = pd.concat(X_split[:i] + X_split[i+1:])
        y_totrain = X_totrain['resultat']
        # On supprime les résultats des données à entraîner pour ne pas les utiliser dans l'entraînement
        X_topred.drop('resultat', axis=1, inplace=True)
        X_totrain.drop('resultat', axis=1, inplace=True)
        # On entraîne l'arbre de décision
        arbre = train(X_totrain, y_totrain, crit = critere)
        # On récupère la précision 
        acc = getAccuracy(arbre, X_topred, y_topred)
        scores.append(acc)
    return scores

def train_test_split(X, y, test_size: float = 0.2):
    """Cette fonction permet de diviser les données en un ensemble d'entraînement et un ensemble de test

    : param  func: la fonction qui permet de récupérer les données
    : param  args: les arguments de func
    : param  test_size: la taille de l'ensemble de test

    : return X_train: les données d'entraînement
    : return X_test: les données de test
    : return y_train: les labels d'entraînement
    : return y_test: les labels de test
    """
    # Shuffle the data
    X, y = shuffle(X, y)
    # Split the data
    X_train = X[:int(len(X) * (1 - test_size))]
    X_test = X[int(len(X) * (1 - test_size)):]
    y_train = y[:int(len(y) * (1 - test_size))]
    y_test = y[int(len(y) * (1 - test_size)):]
    return X_train, X_test, y_train, y_test

def getAccuracy(clf: tree.DecisionTreeClassifier, X_test: list, y_test: list):
    """Cette fonction permet de calculer la précision d'un classifieur

    : param  clf: le classifieur
    : param  X_test: les données de test
    : param  y_test: les labels de test

    : return accuracy: la précision du classifieur
    """
    accuracy = 0
    y_pred = clf.predict(X_test)
    for i in range(len(y_pred)):
        if y_pred[i] == y_test.iloc[i]:
            accuracy += 1
    accuracy /= len(y_pred)
    return accuracy

def traceMatriceConf(clf: tree.DecisionTreeClassifier, X_test: list, y_test: list):
    """ Cette fonction permet de tracer la matrice de confusion d'un classifieur
    
    : param  clf: le classifieur
    : param  X_test: les données de test
    : param  y_test: les vrais valeurs
    """
    y_pred = clf.predict(X_test)
    mat_conf = confusion_matrix(y_test,y_pred)
    fig, ax = plt.subplots()
    M = sns.heatmap(mat_conf/np.sum(mat_conf),annot=True,fmt='.2%',cmap='Blues')
    plt.suptitle('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    ax.set_xticklabels(['red','blue'])
    ax.set_yticklabels(['red','blue'])
    plt.show()

def train(X_train: list, y_train: list, min_samples_split: int=2, max_depth: int=None, crit: str='gini'):
    """Cette fonction permet d'entraîner un classifieur

    : param  X_train: les données d'entraînement
    : param  y_train: les labels d'entraînement
    : param  min_samples_split: le nombre minimum d'échantillons requis pour diviser un noeud
    : param  max_depth: la profondeur maximale de l'arbre
    : param crit: la fonction de mesure de la qualité de la séparation (gini ou entropy)

    : return clf: le classifieur
    """
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, criterion=crit)
    clf = clf.fit(X_train, y_train)
    return clf  # criterion='gini'est par défaut (gini ou entropy)

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
        # print(bestParams)
        accuracy.append([])
        for j in max_depth:
            clf = train(X_train, y_train, i, j)
            accuracy[-1].append(getAccuracy(clf, X_test, y_test))
            print('min_samples_split =', i, 'max_depth =', j, 'accuracy =', accuracy[-1][-1])
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

def classe(tree : tree.DecisionTreeClassifier, newdatas: pd.DataFrame):
    """
    Cette fonction permet d'obtenir la classe d'un nouvel échantillon à partir d'un arbre de décision et d'ajouter le resultat dans le dataframe matches
    :param tree: l'arbre de décision
    :param newdatas: le dataframe contenant les nouvelles données
    :return: le dataframe matches avec la nouvelle colonne result
    """
    result = []
    for i in range(len(newdatas)):
        result.append(tree.predict([newdatas.iloc[i]]))
    newdatas['result'] = result
    return newdatas



if __name__ == '__main__' :

    if 0:
        stats = pd.read_csv('full_stats.csv', index_col=None)
        for color in ['blue', 'red']:
            for role in ['top', 'jungle', 'mid', 'adc', 'support']:
                for stat in ['Fighter', 'Tank', 'Mage', 'Assassin', 'Support', 'Marksman']:
                    stats[color + role + stat] = getStat_labonne(color, role, stat,matches)
        stats['result'] = matches['result']
        stats.to_csv('full_stats.csv', index=False)

    # # Exemple d'utilisation des fonctions
    # X, y = prepare_donnee(getStat_red_blue, ('top', ('hp',)))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # clf = train(X_train, y_train, 100, 5)
    # print(getAccuracy(clf, X_test, y_test)) # 0.5125786163522013, pas super mais mieux que rien

    # # bestParams automatise la recherche des meilleurs paramètres (min_samples_split et max_depth)
    # params = bestParamsplot(getStat_red_blue, ('top', ('hp',)), test_size=0.2)
    # print(params)

    # # bestParamsplot affiche aussi une heatmap des résultats
    # params = bestParamsplot(getStat_red_blue, ('top', ('hp',)), test_size=0.2)
    # print(params)

    # # Les hps seul ne sont pas très efficaces, on peut essayer avec les hps et les mps
    # X, y = prepare_donnee(getStat_red_blue, ('top', ('hp', 'mp')))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # clf = train(X_train, y_train, 100, 5)
    # print(getAccuracy(clf, X_test, y_test)) # 0.5157232704402516, un peu mieux, il ne reste plus qu'à mettre plus de paramètres

    # params = bestParamsplot(getStat_red_blue, ('top', ('hp', 'mp')), test_size=0.2)
    # print(params)

    # # On va essayer avec toutes les stats (ça va probablement être long (effectivement, ça a mis 02h 07min... les meilleurs paramètres sont min_samples_split = 2 max_depth = 5))
    stats_names = ('attack', 'defense', 'magic', 'difficulty', 'Fighter', 'Tank', 'Mage', 'Assassin', 'Support', 'Marksman', 'hp', 'hpperlevel', 'mp', 'mpperlevel', 'movespeed', 'armor', 'armorperlevel', 'spellblock', 'spellblockperlevel', 'attackrange', 'hpregen', 'hpregenperlevel', 'mpregen', 'mpregenperlevel', 'attackdamage', 'attackdamageperlevel', 'attackspeedperlevel', 'attackspeed')
    roles = ('top', 'jungle', 'mid', 'adc', 'support')
    # # params = bestParamsplot(getStat_red_blue, *[(pos, stat) for pos in roles], test_size=0.2)
    # # print(params)

    # On va sauvegarder le meilleur arbre et le dataset pour ne pas les générer à chaque fois
    
    # pickle.dump(X_train, open('full_X_train_difference.pkl', 'wb'))
    # pickle.dump(X_test, open('full_X_test_difference.pkl', 'wb'))
    # pickle.dump(y_train, open('full_y_train_difference.pkl', 'wb'))
    # pickle.dump(y_test, open('full_y_test_difference.pkl', 'wb'))

    # X_train = pickle.load(open('full_X_train_difference.pkl', 'rb'))
    # X_test = pickle.load(open('full_X_test_difference.pkl', 'rb'))
    # y_train = pickle.load(open('full_y_train_difference.pkl', 'rb'))
    # y_test = pickle.load(open('full_y_test_difference.pkl', 'rb'))

    print("****************************************** DIFFERENCE ******************************************")
    X, y = prepare_donnee(getStat_difference, *[(pos, stats_names) for pos in roles])
    scores = cross_validation(X, y, 10)
    print('La liste des scores est donnée par ', scores)
    print('La moyenne des scores est de ', np.mean(scores))
    print('L\'écart type des scores est de ', np.std(scores))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = train(X_train, y_train, 2, 3) # prends ~ 3min
    # pickle.dump(clf, open('full_tree_difference.pkl', 'wb'))# 2 3 <- mettre à jour si on change les paramètres

    # clf = pickle.load(open('full_tree_difference.pkl', 'rb'))
    
    plt.figure()

    tree.plot_tree(clf, feature_names=X.columns, class_names=['red', 'blue'])
    plt.title("Arbre de décision pour la différence des stats")
    print("accuracy difference = ", getAccuracy(clf, X_test, y_test)) # 0.5293501048218029 avec 2 3
    plt.show()
    traceMatriceConf(clf, X_test, y_test)


    print("****************************************** RATIO ******************************************")
    X, y = prepare_donnee(getStat_rapport, *[(pos, stats_names) for pos in roles])
    scores = cross_validation(X, y, 10)
    print('La liste des scores est donnée par ', scores)
    print('La moyenne des scores est de ', np.mean(scores))
    print('L\'écart type des scores est de ', np.std(scores))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = train(X_train, y_train, 2, 3) # prends ~ 3min

    plt.figure()
    tree.plot_tree(clf, feature_names=X.columns, class_names=['red', 'blue'])
    plt.title("Arbre de décision pour le rapport des stats")
    print("accuracy ratio = ", getAccuracy(clf, X_test, y_test)) # 0.5293501048218029 avec 2 3
    plt.show()
    traceMatriceConf(clf, X_test, y_test)
    # params = bestParamsplot(X_train, X_test, y_train, y_test, range(1, 50, 2), range(1, 50, 2))
    # print(params)
    # clf = train(X_train, y_train, 11, 29) # prends ~ 3min
    # # print(matches['bluetop'])
    # # clf = tree.DecisionTreeClassifier(min_samples_split=100, max_depth=5)
    # # clf = clf.fit([[i,j] for i, j in zip(getStats(matches['bluetop'],'hp'), getStats(matches['redtop'],'hp'))], matches['result'])
    # tree.plot_tree(clf)
    # plt.show()
    # # entrainer un arbre a partir de getstat_difference 
    # # train_test_split(getStat_difference, ('top', ('attack',)), test_size=0.5)
    # # clf = tree.DecisionTreeClassifier(min_samples_split=100, max_depth=5)

