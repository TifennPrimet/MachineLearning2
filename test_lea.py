from premiersArbres import *

# Test 1
# On regarde les stats du role 'top' sur 'hp', 'armure' et 'attack'
# On entraine 50% du jeu de données

# On sépare le jeu de données
# X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp','armor','attack')), test_size=0.5)
# On enregistre la séparation
# pickle.dump(X_train, open('pkl_lea/test1_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_lea/test1_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_lea/test1_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_lea/test1_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_lea/test1_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_lea/test1_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_lea/test1_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_lea/test1_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 13 et depth = 11
# On fait l'arbre
# arbre = train(X_train,y_train,13,11)
# pickle.dump(arbre, open('pkl_lea/test1_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_lea/test1_tree.pkl', 'rb'))
tree.plot_tree(arbre)
# On calcule la précision (53.29140461215933)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 1.')

# Test 2
# On regarde les stats du role 'top' sur 'hp', 'armure' et 'attack'
# Cette fois ci, on entraine 80% du jeu de données 

# On sépare le jeu de données
# X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp','armor','attack')), test_size=0.2)
# On enregistre la séparation
# pickle.dump(X_train, open('pkl_lea/test2_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_lea/test2_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_lea/test2_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_lea/test2_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_lea/test2_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_lea/test2_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_lea/test2_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_lea/test2_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 2 et depth = 2
# On fait l'arbre
# arbre = train(X_train,y_train,2,2)
# pickle.dump(arbre, open('pkl_lea/test2_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_lea/test2_tree.pkl', 'rb'))
tree.plot_tree(arbre)
# On calcule la précision (52.6329578202777)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 2.')

# Test 3
# On regarde les stats de l'idée 2
# On entraine 50% des données

# On sépare le jeu de données
# X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')), ('support',('hp',)), test_size=0.5)
# On enregistre la séparation
# pickle.dump(X_train, open('pkl_lea/test3_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_lea/test3_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_lea/test3_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_lea/test3_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_lea/test3_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_lea/test3_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_lea/test3_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_lea/test3_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 17 et depth = 9
# On fait l'arbre
# arbre = train(X_train,y_train,17,9)
# pickle.dump(arbre, open('pkl_lea/test3_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_lea/test3_tree.pkl', 'rb'))
tree.plot_tree(arbre)
# On calcule la précision (0.5316561844863732)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 3.')

# Test 4
# On regarde les stats de l'idée 2
# On entraine 80% des données

# On sépare le jeu de données
X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')), ('support',('hp',)), test_size=0.2)
# On enregistre la séparation
pickle.dump(X_train, open('pkl_lea/test3_X_train.pkl', 'wb'))
pickle.dump(X_test, open('pkl_lea/test3_X_test.pkl', 'wb'))
pickle.dump(y_train, open('pkl_lea/test3_y_train.pkl', 'wb'))
pickle.dump(y_test, open('pkl_lea/test3_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_lea/test3_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_lea/test3_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_lea/test3_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_lea/test3_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 2 et depth = 2
# On fait l'arbre
# arbre = train(X_train,y_train,3,14)
# pickle.dump(arbre, open('pkl_lea/test3_tree.pkl', 'wb'))

# # Test 5 
# # On regarde les tags dans chaque role
# # On entraine 50% des donnees

# # On sépare le jeu de données
# # X_train, X_test, y_train, y_test = train_test_split(getStat_red_blue, ('top', ('Tank', 'Fighter')), ('jungle', ('Fighter',)), ('mid',('Mage',)),('adc',('Marksman',)), ('support',('Support',)), test_size=0.5)
# # On enregistre la séparation
# # pickle.dump(X_train, open('pkl_lea/test3_X_train.pkl', 'wb'))
# # pickle.dump(X_test, open('pkl_lea/test3_X_test.pkl', 'wb'))
# # pickle.dump(y_train, open('pkl_lea/test3_y_train.pkl', 'wb'))
# # pickle.dump(y_test, open('pkl_lea/test3_y_test.pkl', 'wb'))