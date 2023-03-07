from premiersArbres import *
#test 4 :
# On entraine 80% des données

#On sépare le jeu de données
#X, y = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')))
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# On enregistre la séparation
#pickle.dump(X_train, open('pkl_rita/test4_X_train.pkl', 'wb'))
#pickle.dump(X_test, open('pkl_rita/test4_X_test.pkl', 'wb'))
#pickle.dump(y_train, open('pkl_rita/test4_y_train.pkl', 'wb'))
#pickle.dump(y_test, open('pkl_rita/test4_y_test.pkl', 'wb'))

#X_train = pickle.load(open('pkl_rita/test4_X_train.pkl', 'rb'))
#X_test = pickle.load(open('pkl_rita/test4_X_test.pkl', 'rb'))
#y_train = pickle.load(open('pkl_rita/test4_y_train.pkl', 'rb'))
#y_test = pickle.load(open('pkl_rita/test4_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
#best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 70), range(2, 70)) # donne samples = 2 et depth = 2
#print('Le meilleur paramètre =', best)
# On fait l'arbre
#arbre_gini = train(X_train,y_train,19,13,'gini') # samples = 19 et depth = 13
#pickle.dump(arbre_gini, open('pkl_rita/test4_tree.pkl', 'wb'))
#arbre_gini = pickle.load(open('pkl_rita/test4_tree.pkl', 'rb'))
#tree.plot_tree(arbre_gini,feature_names=X.columns,class_names=['red','blue'])
#arbre_entropy = train(X_train,y_train,19,13,'entropy') # samples = 19 et depth = 13
#pickle.dump(arbre_entropy, open('pkl_rita/test4_tree.pkl', 'wb'))
#arbre_entropy = pickle.load(open('pkl_rita/test4_tree.pkl', 'rb'))
#tree.plot_tree(arbre_entropy,feature_names=X.columns,class_names=['red','blue'])
#plt.show()

#acc_gini = getAccuracy(arbre_gini,X_test,y_test)
#acc_entropy = getAccuracy(arbre_entropy,X_test,y_test)
#print('La precision avec gini est de ', acc_gini*100, '% pour le test 4.')
#print('La precision avec entropy est de ', acc_entropy*100, '% pour le test 4.')
#La precision avec gini est de  56.18448637316562 % pour le test 4.
#La precision avec entropy est de 55.34591194968554 % pour le test 4.
#matrice de confusion:
#traceMatriceConf(arbre_gini,X_test,y_test)
#traceMatriceConf(arbre_entropy,X_test,y_test)


#test 5 :
# On entraine 80% des données

#On sépare le jeu de données
#X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)), test_size=0.2)
# On enregistre la séparation
#pickle.dump(X_train, open('pkl_rita/test5_X_train.pkl', 'wb'))
#pickle.dump(X_test, open('pkl_rita/test5_X_test.pkl', 'wb'))
#pickle.dump(y_train, open('pkl_rita/test5_y_train.pkl', 'wb'))
#pickle.dump(y_test, open('pkl_rita/test5_y_test.pkl', 'wb'))

#X_train = pickle.load(open('pkl_rita/test5_X_train.pkl', 'rb'))
#X_test = pickle.load(open('pkl_rita/test5_X_test.pkl', 'rb'))
#y_train = pickle.load(open('pkl_rita/test5_y_train.pkl', 'rb'))
#y_test = pickle.load(open('pkl_rita/test5_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
#best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 60), range(2, 60)) # donne samples = 2 et depth = 2
#print('Le meilleur paramètre =', best)
# On fait l'arbre
#arbre_gini = train(X_train,y_train,7,26,'gini') # samples = 7 et depth = 26
#pickle.dump(arbre_gini, open('pkl_rita/test5_tree.pkl', 'wb'))
#arbre_gini = pickle.load(open('pkl_rita/test5_tree.pkl', 'rb'))
#tree.plot_tree(arbre_gini,feature_names=X.columns,class_names=['red','blue'])
#arbre_entropy = train(X_train,y_train,7,26,'entropy') # samples = 7 et depth = 26
#pickle.dump(arbre_entropy, open('pkl_rita/test5_tree.pkl', 'wb'))
#arbre_entropy = pickle.load(open('pkl_rita/test5_tree.pkl', 'rb'))
#tree.plot_tree(arbre_entropy,feature_names=X.columns,class_names=['red','blue'])
#plt.show()
# On calcule la précision (0.5461215932914046)
#acc_gini = getAccuracy(arbre_gini,X_test,y_test)
#acc_entropy = getAccuracy(arbre_entropy,X_test,y_test)
#print('La precision avec gini est de ', acc_gini*100, '% pour le test 5.')
#print('La precision avec entropy est de ', acc_entropy*100, '% pour le test 5.')
# La precision avec gini est de 54.29769392033543 % pour le test 5 avec gini.
# La precision avec entropy est de 51.36268343815513 % pour le test 5 avec entropy.
#matrice de confusion:
#traceMatriceConf(arbre_gini,X_test,y_test)
#traceMatriceConf(arbre_entropy,X_test,y_test)

#test 6 :
# On entraine 80% des données

#On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack','movespeed',)), ('mid',('magic',)), ('support',('hp',)))
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# On enregistre la séparation
#pickle.dump(X_train, open('pkl_rita/test6_X_train.pkl', 'wb'))
#pickle.dump(X_test, open('pkl_rita/test6_X_test.pkl', 'wb'))
#pickle.dump(y_train, open('pkl_rita/test6_y_train.pkl', 'wb'))
#pickle.dump(y_test, open('pkl_rita/test6_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_rita/test6_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_rita/test6_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_rita/test6_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_rita/test6_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
#best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 45), range(2, 45)) # donne samples = 2 et depth = 2
#print('Le meilleur paramètre =', best)
# On fait l'arbre
arbre_gini = train(X_train,y_train,2,6,'gini') # samples = 2 et depth = 6
pickle.dump(arbre_gini, open('pkl_rita/test6_tree.pkl', 'wb'))
arbre_gini = pickle.load(open('pkl_rita/test6_tree.pkl', 'rb'))
tree.plot_tree(arbre_gini,feature_names=X.columns,class_names=['red','blue'])
arbre_entropy = train(X_train,y_train,2,6,'entropy') # samples = 2 et depth = 6
pickle.dump(arbre_entropy, open('pkl_rita/test6_tree.pkl', 'wb'))
arbre_entropy = pickle.load(open('pkl_rita/test6_tree.pkl', 'rb'))
tree.plot_tree(arbre_entropy,feature_names=X.columns,class_names=['red','blue'])
plt.show()

acc_gini = getAccuracy(arbre_gini,X_test,y_test)
acc_entropy = getAccuracy(arbre_entropy,X_test,y_test)
print('La precision avec gini est de ', acc_gini*100, '% pour le test 6.')
print('La precision avec entropy est de ', acc_entropy*100, '% pour le test 6.')
# La precision avec gini est de  55.34591194968554 % pour le test 6 avec gini.
# La precision avec entropy est de 54.088050314465406 % pour le test 6 avec entropy.
#matrice de confusion:
traceMatriceConf(arbre_gini,X_test,y_test)
traceMatriceConf(arbre_entropy,X_test,y_test)


