from premiersArbres import *
#test 4 :
# On entraine 80% des données

#On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# On enregistre la séparation
#pickle.dump(X_train, open('pkl_rita/test4_X_train.pkl', 'wb'))
#pickle.dump(X_test, open('pkl_rita/test4_X_test.pkl', 'wb'))
#pickle.dump(y_train, open('pkl_rita/test4_y_train.pkl', 'wb'))
#pickle.dump(y_test, open('pkl_rita/test4_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_rita/test4_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_rita/test4_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_rita/test4_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_rita/test4_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
#best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 70), range(2, 70)) # donne samples = 2 et depth = 2
#print('Le meilleur paramètre =', best)
# On fait l'arbre
#arbre = train(X_train,y_train,19,13) # samples = 19 et depth = 13
#pickle.dump(arbre, open('pkl_rita/test4_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_rita/test4_tree.pkl', 'rb'))
tree.plot_tree(arbre)
# On calcule la précision (0.5681341719077568)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 4.')
# La precision est de  56.81341719077568 % pour le test 4.

#test 5 :
# On entraine 80% des données

#On sépare le jeu de données
#X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)), test_size=0.2)
# On enregistre la séparation
#pickle.dump(X_train, open('pkl_rita/test5_X_train.pkl', 'wb'))
#pickle.dump(X_test, open('pkl_rita/test5_X_test.pkl', 'wb'))
#pickle.dump(y_train, open('pkl_rita/test5_y_train.pkl', 'wb'))
#pickle.dump(y_test, open('pkl_rita/test5_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_rita/test5_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_rita/test5_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_rita/test5_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_rita/test5_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
#best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 60), range(2, 60)) # donne samples = 2 et depth = 2
#print('Le meilleur paramètre =', best)
# On fait l'arbre
#arbre = train(X_train,y_train,7,26) # samples = 7 et depth = 26
#pickle.dump(arbre, open('pkl_rita/test5_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_rita/test5_tree.pkl', 'rb'))
tree.plot_tree(arbre)
# On calcule la précision (0.5461215932914046)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 5.')
# La precision est de  54.61215932914046 % pour le test 5.

