from premiersArbres import *

# =============================================================================
# Idée 2

    # {top} : 'hp', 'armure', 'attack' 
    # {jungle} : 'attack', 'movespeed' 
    # {mid} : dégâts magiques (ne doit pas mourir) 
    # {adc} : (ne doit pas mourir) 'attack', 'attackspeed', 'movespeed' 
    # {support} : 'hp' (dur à prévoir sans item)

# =============================================================================

# Test 1
print('##### Test 1 #####')
# On regarde les stats du role 'top' sur 'hp', 'armure' et 'attack'
# On entraine 50% du jeu de données

# On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/test1_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/test1_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/test1_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/test1_y_test.pkl', 'wb'))

# Validation du modèle
scores = cross_validation(X, y, 10)
print('La liste des scores est donnée par ', scores)
print('La moyenne des scores est de ', np.mean(scores))
print('L\'écart type des scores est de ', np.std(scores))

X_train = pickle.load(open('pkl_arbrestats/test1_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/test1_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/test1_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/test1_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 13 et depth = 11
# On fait l'arbre
arbre = train(X_train,y_train,13,11)
pickle.dump(arbre, open('pkl_arbrestats/test1_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_arbrestats/test1_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre,feature_names=X.columns,class_names=['red','blue'])
# On calcule la précision (53.29140461215933)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 1.')
# Matrice de confusion
traceMatriceConf(arbre, X_test, y_test)

# Test 2
print('\n##### Test 2 #####')
# On regarde les stats du role 'top' sur 'hp', 'armure' et 'attack'
# Cette fois ci, on entraine 80% du jeu de données 

# On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/test2_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/test2_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/test2_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/test2_y_test.pkl', 'wb'))

# Validation du modèle
scores = cross_validation(X, y, 10)
print('La liste des scores est donnée par ', scores)
print('La moyenne des scores est de ', np.mean(scores))
print('L\'écart type des scores est de ', np.std(scores))

X_train = pickle.load(open('pkl_arbrestats/test2_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/test2_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/test2_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/test2_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 2 et depth = 2
# On fait l'arbre
arbre = train(X_train,y_train,2,2)
pickle.dump(arbre, open('pkl_arbrestats/test2_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_arbrestats/test2_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre,feature_names=X.columns,class_names=['red','blue'])
# On calcule la précision (52.09643605870021)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 2.')
# Matrice de confusion
traceMatriceConf(arbre, X_test, y_test)

# Test 3
print('\n##### Test 3 #####')
# On regarde les stats de l'idée 2
# On entraine 50% des données

# On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')), ('support',('hp',)))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/test3_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/test3_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/test3_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/test3_y_test.pkl', 'wb'))

# Validation du modèle
scores = cross_validation(X, y, 10)
print('La liste des scores est donnée par ', scores)
print('La moyenne des scores est de ', np.mean(scores))
print('L\'écart type des scores est de ', np.std(scores))

X_train = pickle.load(open('pkl_arbrestats/test3_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/test3_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/test3_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/test3_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 17 et depth = 9
# On fait l'arbre
arbre = train(X_train,y_train,17,9)
pickle.dump(arbre, open('pkl_arbrestats/test3_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_arbrestats/test3_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre,feature_names=X.columns,class_names=['red','blue'])
# On calcule la précision (54.17190775681342)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 3.')
# Matrice de confusion
traceMatriceConf(arbre, X_test, y_test)

# Test 4
print('\n##### Test 4 #####')
# On regarde les stats de l'idée 2
# On entraine 80% des données

# On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')), ('support',('hp',)))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/test4_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/test4_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/test4_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/test4_y_test.pkl', 'wb'))

# Validation du modèle
scores = cross_validation(X, y, 10)
print('La liste des scores est donnée par ', scores)
print('La moyenne des scores est de ', np.mean(scores))
print('L\'écart type des scores est de ', np.std(scores))

X_train = pickle.load(open('pkl_arbrestats/test4_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/test4_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/test4_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/test4_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 2 et depth = 14
# On fait l'arbre
arbre = train(X_train,y_train,2,14)
pickle.dump(arbre, open('pkl_arbrestats/test4_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_arbrestats/test4_tree.pkl', 'rb'))
fig, ax = plt.subplots()
plt.figure()
tree.plot_tree(arbre,feature_names=X.columns,class_names=['red','blue'])
ax.set_title('Arbre de décision pour le test 4')
plt.savefig('tree.pgf')
# On calcule la précision (54.19287211740041)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 4.')
# Matrice de confusion
traceMatriceConf(arbre, X_test, y_test)


# Test 5 
print('\n##### Test 5 #####')
# On regarde les tags dans chaque role
# On entraine 50% des donnees

# On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('Tank', 'Fighter')), ('jungle', ('Fighter',)), ('mid',('Mage',)),('adc',('Marksman',)), ('support',('Support',)))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/test5_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/test5_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/test5_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/test5_y_test.pkl', 'wb'))

scores = cross_validation(X, y, 10)
print('La liste des scores est donnée par ', scores)
print('La moyenne des scores est de ', np.mean(scores))
print('L\'écart type des scores est de ', np.std(scores))

X_train = pickle.load(open('pkl_arbrestats/test5_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/test5_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/test5_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/test5_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 20 et depth = 8
# On fait l'arbre
arbre = train(X_train,y_train,20,8)
pickle.dump(arbre, open('pkl_arbrestats/test5_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_arbrestats/test5_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre,feature_names=X.columns,class_names=['red','blue'])
# On calcule la précision (53.9622641509434)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 5.')
# Matrice de confusion
traceMatriceConf(arbre, X_test, y_test)


# Test 6
print('\n##### Test 6 #####')
# On regarde les tags dans chaque role
# On entraine 80% des donnees

# On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('Tank', 'Fighter')), ('jungle', ('Fighter',)), ('mid',('Mage',)),('adc',('Marksman',)), ('support',('Support',)))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/test6_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/test6_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/test6_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/test6_y_test.pkl', 'wb'))

scores = cross_validation(X, y, 10)
print('La liste des scores est donnée par ', scores)
print('La moyenne des scores est de ', np.mean(scores))
print('L\'écart type des scores est de ', np.std(scores))

X_train = pickle.load(open('pkl_arbrestats/test6_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/test6_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/test6_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/test6_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 14 et depth = 5
# On fait l'arbre
arbre = train(X_train,y_train,13,5)
pickle.dump(arbre, open('pkl_arbrestats/test6_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_arbrestats/test6_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre,feature_names=X.columns,class_names=['red','blue'])
# On calcule la précision (52.868745087765255)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 6.')
# Matrice de confusion
traceMatriceConf(arbre, X_test, y_test)

# Test 7
print('\n##### Test 7 #####')
# On regarde les tags dans chaque role
# On entraine 80% des donnees

# On sépare le jeu de données
X, y = prepare_donnee(getStat_red_blue, ('top', ('Tank', 'Fighter','hp','armor','attack')), ('jungle', ('Fighter','attack','movespeed')), ('mid',('Mage','magic')),('adc',('Marksman','attack','attackspeed','movespeed')), ('support',('Support','hp')))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/test7_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/test7_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/test7_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/test7_y_test.pkl', 'wb'))

scores = cross_validation(X, y, 10)
print('La liste des scores est donnée par ', scores)
print('La moyenne des scores est de ', np.mean(scores))
print('L\'écart type des scores est de ', np.std(scores))

X_train = pickle.load(open('pkl_arbrestats/test7_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/test7_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/test7_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/test7_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 14 et depth = 5
# On fait l'arbre
arbre = train(X_train,y_train,7,4)
pickle.dump(arbre, open('pkl_arbrestats/test7_tree.pkl', 'wb'))
arbre = pickle.load(open('pkl_arbrestats/test7_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre,feature_names=X.columns,class_names=['red','blue'])
# On calcule la précision (52.947340843594446)
acc = getAccuracy(arbre,X_test,y_test)
print('La precision est de ', acc*100, '% pour le test 7.')
# Matrice de confusion
traceMatriceConf(arbre, X_test, y_test)


# =============================================================================
# Prediction sur les nouvelles données
# =============================================================================

# Test 8 
print('\n##### Test 8 #####')
# On construit un arbre sur toutes les données basé sur l'idée 2
X2complet, y2complet = prepare_donnee(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')), ('support',('hp',)))
# X_train, X_test, y_train, y_test = train_test_split(X2complet, y2complet, test_size=0.0)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/complet2_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/complet2_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/complet2_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/complet2_y_test.pkl', 'wb'))


X_train = pickle.load(open('pkl_arbrestats/complet2_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/complet2_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/complet2_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/complet2_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 14 et depth = 5
# On fait l'arbre
arbre_complet2 = train(X_train,y_train,10,10)
pickle.dump(arbre_complet2, open('pkl_arbrestats/complet2_tree.pkl', 'wb'))
arbre_complet2 = pickle.load(open('pkl_arbrestats/complet2_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre_complet2,feature_names=X2complet.columns,class_names=['red','blue'])

# Nouvelles données
X_predire2 = prepare_new_donnee(getStat_red_blue, ('top', ('hp','armor','attack')), ('jungle', ('attack',)), ('mid',('magic',)),('adc',('attack','attackspeed')), ('support',('hp',)))
prediction2 = classe(arbre_complet2,X_predire2.copy())['result']
print('La prédiction est', prediction2)

# Test 9
print('\n##### Test 9 #####')
# On construit un arbre sur toutes les données basé sur l'idée 2
Xcomplet_champ, ycomplet_champ = prepare_donnee(getStat_red_blue, ('top', ('Tank', 'Fighter')), ('jungle', ('Fighter',)), ('mid',('Mage',)),('adc',('Marksman',)), ('support',('Support',)))
# X_train, X_test, y_train, y_test = train_test_split(Xcomplet_champ, ycomplet_champ, test_size=0.0)
# # On enregistre la séparation
# pickle.dump(X_train, open('pkl_arbrestats/completchamp_X_train.pkl', 'wb'))
# pickle.dump(X_test, open('pkl_arbrestats/completchamp_X_test.pkl', 'wb'))
# pickle.dump(y_train, open('pkl_arbrestats/completchamp_y_train.pkl', 'wb'))
# pickle.dump(y_test, open('pkl_arbrestats/completchamp_y_test.pkl', 'wb'))


X_train = pickle.load(open('pkl_arbrestats/completchamp_X_train.pkl', 'rb'))
X_test = pickle.load(open('pkl_arbrestats/completchamp_X_test.pkl', 'rb'))
y_train = pickle.load(open('pkl_arbrestats/completchamp_y_train.pkl', 'rb'))
y_test = pickle.load(open('pkl_arbrestats/completchamp_y_test.pkl', 'rb'))

# On calcule les meilleurs paramètres
# best = bestParamsplot(X_train, X_test, y_train, y_test, range(2, 30), range(2, 30)) # donne samples = 14 et depth = 5
# On fait l'arbre
arbre_completchamp = train(X_train,y_train,10,10)
pickle.dump(arbre_completchamp, open('pkl_arbrestats/completchamp_tree.pkl', 'wb'))
arbre_completchamp = pickle.load(open('pkl_arbrestats/completchamp_tree.pkl', 'rb'))
plt.figure()
tree.plot_tree(arbre_completchamp,feature_names=Xcomplet_champ.columns,class_names=['red','blue'])

# Nouvelles données
X_predire_champ = prepare_new_donnee(getStat_red_blue, ('top', ('Tank', 'Fighter')), ('jungle', ('Fighter',)), ('mid',('Mage',)),('adc',('Marksman',)), ('support',('Support',)))
prediction_champ = classe(arbre_completchamp,X_predire_champ.copy())['result']
print('La prédiction est', prediction_champ)

plt.show()