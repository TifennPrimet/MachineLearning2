import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.utils import shuffle
from premiersArbres import getStat, getStat_red_blue, getStat_difference, getStat_rapport, prepare_donnee, train_test_split, getAccuracy, train, bestParamsplot

if 1==1: # Lecture des données
    champion = pd.read_csv('champions.csv', index_col=None)
    # Les tags sont encore sous forme de chaîne de caractères, nous devons les convertir en listes
    champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
    matches = pd.read_csv('matches.csv', index_col=None)
    champion.drop("crit", axis=1, inplace=True)
    champion.drop("critperlevel", axis=1, inplace=True)
    print(matches.head())
    print( " I got the datas ! ")


def cree_list_leagues(data):
    """
    Crée un dataframe de toutes les equipes présentes dans le dataset data ainsi 
    qu'un colonne avec le nombre de matchs joués dans chaque equipe et un colonne 
    avec le nombre de matchs gagnés dans chaque equipe
    """
    # On crée une liste de toutes les equipes présentes dans le dataset
    list_equipes = []
    for i in range(len(data)):
        if data['blueteam'][i] not in list_equipes:
            list_equipes.append(data['blueteam'][i])
        if data['redteam'][i] not in list_equipes:
            list_equipes.append(data['redteam'][i])
    # On crée un dataframe avec les equipes et on ajoute une colonne avec le nombre de matchs joués
    df = pd.DataFrame(list_equipes, columns=['equipe'])
    df['nb_matchs'] = 0
    df['nb_victoires'] = 0
    # On ajoute le nombre de matchs joués dans chaque equipe
    for i in range(len(data)):
        df.loc[df['equipe'] == data['blueteam'][i], 'nb_matchs'] += 1
        df.loc[df['equipe'] == data['redteam'][i], 'nb_matchs'] += 1
    # On ajoute le nombre de matchs gagnés dans chaque equipe
    for i in range(len(data)):
        if data['result'][i] == 1:
            df.loc[df['equipe'] == data['blueteam'][i], 'nb_victoires'] += 1
        else:
            df.loc[df['equipe'] == data['redteam'][i], 'nb_victoires'] += 1
    return df

def ratio_victoire(data):
    """
    Ajoute une colonne au dataframe data avec le ratio de victoire de chaque equipe
    """
    data['ratio_victoire'] = data['nb_victoires']/data['nb_matchs']
    return data

def ajoute_ratio_victoire(data):
    """
    Ajoute une colonne au dataframe data avec le ratio de victoire de chaque equipe
    """
    victoire = ratio_victoire(cree_list_leagues(data))
    # On ajoute le ratio de victoire de l'equipe bleu et de l'equipe bleu dans le dataframe data
    data['ratio_victoire_blueteam'] = 0
    data['ratio_victoire_redteam'] = 0
    for i in range(len(data)):
        data.loc[i, 'ratio_victoire_blueteam'] = victoire.loc[victoire['equipe'] == data['blueteam'][i], 'ratio_victoire'].values
        data.loc[i, 'ratio_victoire_redteam'] = victoire.loc[victoire['equipe'] == data['redteam'][i], 'ratio_victoire'].values

    # On ajoute la différence de ratio de victoire entre les deux equipes
    data['diff_ratio_victoire'] = data['ratio_victoire_blueteam'] - data['ratio_victoire_redteam']
    return data

def ajout_full_stat( name,data):
    """ 
    On ajoute les differents ratio et differences de chaque equipe dans le fichier full_stat.csv
    """
    data_avec_ratio = ajoute_ratio_victoire(data)
    stats = pd.read_csv(name, index_col=None)
    stats['ratio_victoire_blueteam'] = data_avec_ratio['ratio_victoire_blueteam']
    stats['ratio_victoire_redteam'] = data_avec_ratio['ratio_victoire_redteam']
    stats['diff_ratio_victoire'] = data_avec_ratio['diff_ratio_victoire']
    stats.to_csv(name, index=False)

def getStat_ratio_victoire(color, nimportequoi):
    """
    On renvoie la liste des colonnes qui nous interessent pour le ratio de victoire
    """
    return [f"ratio_victoire_{color}team"]

def getStat_difference_ratio_victoire(nimportequoi, nimportequoi2):
    """
    On renvoie la liste des colonnes qui nous interessent pour la difference de ratio de victoire
    """
    return ["diff_ratio_victoire"]

if __name__ == '__main__' :
    ajout_full_stat('full_stats.csv',matches)
    # on lit les donnees de full_stats.csv
    stats = pd.read_csv('full_stats.csv', index_col=None)
    print(stats.head())

    # prend juste les ratio de victoire et la difference de ratio de victoire
    data, result = prepare_donnee(getStat_ratio_victoire, ('blue', ('nimportequoi',)), ('red', ('nimportequoi',)))
    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2)
    clf = train(X_train, y_train, 2, 3) # prends ~ 3min
    fig, ax = plt.subplots()
    tree.plot_tree(clf, feature_names = data.columns, class_names=['red', 'blue'])
    plt.savefig('Tifenn/tree_ratio_victoire.pgf')
    print("accuracy = ", getAccuracy(clf, X_test, y_test)) # 0.5293501048218029 avec 2 3
    plt.show()

    # prend juste la difference de ratio de victoire
    data2, result2 = prepare_donnee(getStat_difference_ratio_victoire, ('nimportequoi', ('nimportequoi2',)))
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data2, result2, test_size=0.2)
    clf2 = train(X_train2, y_train2, 2, 3) # prends ~ 3min
    fig, ax = plt.subplots()
    tree.plot_tree(clf2, feature_names = data.columns, class_names=['red', 'blue'])
    plt.savefig('Tifenn/tree_ratio_victoire_par_equipe.pgf')
    print("accuracy = ", getAccuracy(clf2, X_test2, y_test2)) # 0.5293501048218029 avec 2 3
    plt.show()
