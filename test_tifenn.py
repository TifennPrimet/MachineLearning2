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
    print(champion.head())
    champion.drop("crit", axis=1, inplace=True)
    champion.drop("critperlevel", axis=1, inplace=True)
    print(champion.head())
    print( " I got the datas ! ")
