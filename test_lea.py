from premiersArbres import *

# Test 1
# On regarde les stats du role 'top' sur 'hp', 'armure' et 'attack'
# On entraine 50% du jeu de données

# # On sépare le jeu de données
# X_train, X_test, y_train, y_test = train_test_split(getStat, ('top', ('hp','armor','attack')), test_size=0.5)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_lea/test1_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_lea/test1_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_lea/test1_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_lea/test1_y_test.pkl', 'wb'))

X_train = pickle.load(open('pkl_lea/test1_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_lea/test1_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_lea/test1_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_lea/test1_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 50), range(2, 50)) # donne samples = 12 et depth = 11
# On fait l'arbre
# arbre = train(X_train,y_train,12,11)
# pickle.dump(arbre, open('pkl_lea/test1_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_lea/test1_tree.pkl', 'rb'))
tree.plot_tree(arbre)
# On calcule la précision (0.5316561844863732)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 1.')

# Test 2
# On regarde les stats du role 'top' sur 'hp', 'armure' et 'attack'
# Cette fois ci, on entraine 80% du jeu de données 

# On sépare le jeu de données
X_train, X_test, y_train, y_test = train_test_split(getStat, ('top', ('hp','armor','attack')), test_size=0.8)
# On enregistre la séparation
pickle.dump(X_train, open('pkl_lea/test2_X_train.pkl', 'wb'))
pickle.dump(X_test, open('pkl_lea/test2_X_test.pkl', 'wb'))
pickle.dump(y_train, open('pkl_lea/test2_y_train.pkl', 'wb'))
pickle.dump(y_test, open('pkl_lea/test2_y_test.pkl', 'wb'))

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
# On calcule la précision (0.5316561844863732)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 2.')

