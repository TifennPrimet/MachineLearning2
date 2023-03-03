import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

affichages = { # Modifier les valeurs suivantes pour afficher ou non les plots et les print
    'champion': False, # Nécessaire pour les quatre suivants
    'champion_plot': True,
    'champion_attaque_magie': True,
    'champion_moyenne_ecart_type': True,
    'champion_describe': True,
    'tags': True,
    'tags_roles': True,
    'tags_stats': False, # Nécessaire pour les trois suivants
    'tags_stats_moy': True,
    'tags_stats_std': True,
    'tags_stats_rel_std': True,
    'tags_heatmaps': False, # Nécessaire pour les trois suivants
    'tags_heatmap_champ': True,
    'tags_heatmap_match': True,
    'tags_heatmap_match_rel': True,
    'champions_match': True,
    'champions_match_non_used': True,
    'champions_match_victoire': True,
    'champions_match_defaite': True,
    'champions_match_popularite': True,
    'champions_match_taux_victoire': True,
    'champions_match_taux_defaite': True,
    'champions_match_par_roles': True, # Affichage des statistiques champion-match par rôle (popularité, victoires, défaites)
    'champions_match_par_roles_top': True, # rôle = top
    'champions_match_par_roles_jungle': True, # rôle = jungle
    'champions_match_par_roles_mid': True, # rôle = mid
    'champions_match_par_roles_adc': True, # rôle = adc
    'champions_match_par_roles_support': True, # rôle = support
}

if True: # Lecture des données
    # Lire les données
    champion = pd.read_csv('champions.csv', index_col=None)
    # Les tags sont encore sous forme de chaîne de caractères, nous devons les convertir en listes
    champion['tags'] = champion['tags'].apply(lambda x: x.strip('[]').split(', '))
    matches = pd.read_csv('matches.csv', index_col=None)

if affichages['champion']: # Affichage des données sur les champions
    # Afficher les 5 premières lignes des données sur les champions
    print(champion.head())

    if affichages['champion_plot']:
        # Tracer les données
        champion.plot() # Pas très utile

    if affichages['champion_attaque_magie']:
        plt.figure()
        plt.plot(champion['attack'], champion['magic'], 'o') # Pas très utile non plus
        plt.xlabel('attaque')
        plt.ylabel('magie')

    if affichages['champion_moyenne_ecart_type']:
        # Tracer la moyenne et l'écart type des colonnes numériques (il y en a beaucoup, donc nous devons mettre la légende de l'axe des x de côté)
        champion.plot(kind='box', title='Moyenne et écart type des colonnes numériques') # Pas très utile
        plt.xticks(rotation=90)
        plt.tight_layout()

    if affichages['champion_describe']:
        # Afficher la moyenne et l'écart type des colonnes numériques
        print(champion.describe())

if True: # Obtention des tags
    # Essayons de voir comment les statistiques sont corrélées aux tags
    # Tout d'abord, regardons les tags
    # Les tags sont stockés sous forme de liste de chaînes de caractères, obtenons une liste de tous les tags
    tags = []
    for tag in champion['tags']:
        tags.extend(tag)

if affichages['tags']: # Plot des champions par tag
    # Montrons maintenant le nombre de champions avec chaque tag
    plt.figure("Nombre de champions avec chaque tag")
    sns.histplot(tags, shrink=0.8, color='cornflowerblue')
    plt.xlabel('Tag')
    plt.ylabel('Nombre de champions')
    plt.title('Nombre de champions pour chaque tag')
    plt.tight_layout()

if True: # Calcul des statistiques tag-rôle
    # Montrons la corrélation entre les tags et les rôles
    # Les rôles sont "bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport", "redtop", "redjungle", "redmid", "redadc", "redsupport" et sont des colonnes dans les données de match
    # Tout d'abord, obtenons une liste de tous les rôles
    roles = matches.columns[4:-1]

    # Maintenant, obtenons une liste de tous les tags uniques
    unique_tags = list(set(tags))
    # Maintenant, créons un dataframe avec le nombre de champions ayant chaque tag pour chaque rôle
    tag_role = pd.DataFrame(index=unique_tags, columns=roles)
    for role in roles:
        for tag in unique_tags:
            tag_role.loc[tag, role] = sum(matches[role].isin(champion[champion['tags'].apply(lambda x: tag in x)]['id']))

    # Regroupons les rôles bleus et rouges
    tag_role['top'] = tag_role['bluetop'] + tag_role['redtop']
    tag_role['jungle'] = tag_role['bluejungle'] + tag_role['redjungle']
    tag_role['mid'] = tag_role['bluemid'] + tag_role['redmid']
    tag_role['adc'] = tag_role['blueadc'] + tag_role['redadc']
    tag_role['support'] = tag_role['bluesupport'] + tag_role['redsupport']
    tag_role = tag_role.drop(['bluetop', 'bluejungle', 'bluemid', 'blueadc', 'bluesupport', 'redtop', 'redjungle', 'redmid', 'redadc', 'redsupport'], axis=1)
    # Renverser le dataframe pour que les rôles soient les lignes et les tags les colonnes
    tag_role = tag_role.transpose()

if affichages['tags_roles']: # Plot des tags par rôle
    # Tracer le résultat
    tag_role.plot(kind='bar', stacked=True, title='Nombre de champions avec chaque tag pour chaque rôle')
    plt.xlabel('Rôle')
    plt.ylabel('Nombre de champions')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

if True: # Calcul des statistiques tag-stat (moyenne, écart type, écart type relatif)
    # Montrons maintenant la corrélation entre les tags et les statistiques
    # Les stats sont attack, defense, magic, difficulty, hp, hpperlevel, mp, mpperlevel, movespeed, armor, armorperlevel, spellblock, spellblockperlevel, attackrange, hpregen, hpregenperlevel, mpregen, mpregenperlevel, crit, critperlevel, attackdamage, attackdamageperlevel, attackspeedperlevel, attackspeed et sont des colonnes numériques dans les données des champions
    # On a déjà la liste des tags uniques
    # Créons un dataframe avec la moyenne des statistiques pour chaque tag
    tag_stat_mean = pd.DataFrame(index=unique_tags, columns=champion.columns[2:]) # Contient aussi la colonne tags, on va devoir l'enlever
    tag_stat_mean = tag_stat_mean.drop('tags', axis=1)
    for tag in unique_tags:
        tag_stat_mean.loc[tag] = champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['mean']
    # Renverser le dataframe pour que les statistiques soient les lignes et les tags les colonnes
    tag_stat_mean = tag_stat_mean.transpose()

    # Créons un dataframe avec l'écart type des statistiques pour chaque tag
    tag_stat_std = pd.DataFrame(index=unique_tags, columns=champion.columns[2:]) # Contient aussi la colonne tags, on va devoir l'enlever
    tag_stat_std = tag_stat_std.drop('tags', axis=1)
    for tag in unique_tags:
        tag_stat_std.loc[tag] = champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['std']
    # Renverser le dataframe pour que les statistiques soient les lignes et les tags les colonnes
    tag_stat_std = tag_stat_std.transpose()

    # Créons un dataframe avec l'écart type relatif des statistiques pour chaque tag
    tag_stat_rel_std = pd.DataFrame(index=unique_tags, columns=champion.columns[2:]) # Contient aussi la colonne tags, on va devoir l'enlever
    tag_stat_rel_std = tag_stat_rel_std.drop('tags', axis=1)
    for tag in unique_tags:
        tag_stat_rel_std.loc[tag] = champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['std'] / champion[champion['tags'].apply(lambda x: tag in x)].describe().loc['mean']
    # Renverser le dataframe pour que les statistiques soient les lignes et les tags les colonnes
    tag_stat_rel_std = tag_stat_rel_std.transpose()

if affichages['tags_stats']: # Plot des statistiques tag-stat (moyenne, écart type, écart type relatif)
    if affichages['tags_stats_moy']:
        # Tracer le résultat
        tag_stat_mean.plot(kind='bar', title='Moyenne des statistiques pour chaque tag') # Pas très utile, les moyennes dépendent de la stats mais pas vraiment du tag
        plt.xlabel('Tag')
        plt.ylabel('Moyenne')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    if affichages['tags_stats_std']:
        # Tracer le résultat
        tag_stat_std.plot(kind='bar', title='Ecart type des statistiques pour chaque tag') # Pas très utile, les écart types dépendent trop de la moyenne
        plt.xlabel('Tag')
        plt.ylabel('Ecart type')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    if affichages['tags_stats_rel_std']:
        # Tracer le résultat
        tag_stat_rel_std.plot(kind='bar', title='Ecart type relatif des statistiques pour chaque tag') # Plus utile
        plt.xlabel('Tag')
        plt.ylabel('Ecart type relatif')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

if True: # Calcul des statistiques tag-tag (champions, matchs, relatif)
    # Faisons une heatmap pour voir quels tags sont le plus souvent ensemble
    tag_tag = pd.DataFrame(index=unique_tags, columns=unique_tags, dtype=int)
    for tag1 in unique_tags:
        for tag2 in unique_tags:
            tag_tag.loc[tag1, tag2] = sum(champion['tags'].apply(lambda x: tag1 in x and tag2 in x if tag1 != tag2 else tag1 in x and len(x) == 1))

    # Faisons une heatmap pour voir quels tags sont le plus souvent ensemble dans les matchs
    tag_tag_m = pd.DataFrame(index=unique_tags, columns=unique_tags, dtype=int, data=0)
    for tag1 in unique_tags:
        for tag2 in unique_tags:
            for champ in champion[champion['tags'].apply(lambda x: tag1 in x and tag2 in x if tag1 != tag2 else tag1 in x and len(x) == 1)]['id']:
                # Désolé pour la ligne suivante...
                tag_tag_m.loc[tag1, tag2] += sum(matches['bluetop'] == champ) + sum(matches['bluejungle'] == champ) + sum(matches['bluemid'] == champ) + sum(matches['blueadc'] == champ) + sum(matches['bluesupport'] == champ) + sum(matches['redtop'] == champ) + sum(matches['redjungle'] == champ) + sum(matches['redmid'] == champ) + sum(matches['redadc'] == champ) + sum(matches['redsupport'] == champ)

    # Faisons une heatmap pour voir quelles combinaisons de tags sont les plus populaires
    # On va diviser tag_tag_m par tag_tag
    tag_tag_ = tag_tag_m / tag_tag # Si quelqu'un a une meilleure idée pour tous ces noms, je suis preneur

if affichages['tags_heatmaps']: # Plot des heatmaps tag-tag (champions, matchs, relatif)
    if affichages['tags_heatmap_champ']:
        # Tracer le résultat
        plt.figure("Heatmap des tags")
        sns.heatmap(tag_tag, annot=True, fmt='g', cmap='Blues')
        plt.title('Heatmap des tags')
        plt.xlabel('Tag 1')
        plt.ylabel('Tag 2')
        plt.tight_layout()

    if affichages['tags_heatmap_match']:
        # Tracer le résultat
        plt.figure("Heatmap des tags dans les matchs")
        sns.heatmap(tag_tag_m, annot=True, fmt='g', cmap='Blues')
        plt.title('Heatmap des tags dans les matchs')
        plt.xlabel('Tag 1')
        plt.ylabel('Tag 2')
        plt.tight_layout()

    if affichages['tags_heatmap_match_rel']:
        # Tracer le résultat
        plt.figure("Heatmap des tags dans les matchs normalisée")
        sns.heatmap(tag_tag_, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Heatmap des tags dans les matchs normalisée')
        plt.xlabel('Tag 1')
        plt.ylabel('Tag 2')
        plt.tight_layout()

if True: # Calcul des statistiques champion-match (popularité, victoires, défaites, taux de victoire, taux de défaite)
    # On va maintenant s'intéresser aux matchs : quels champions sont les plus populaires ? Quels sont les champions les plus efficaces ? Quels sont les champions les plus efficaces par rapport à leur popularité ?
    # On va commencer par les champions les plus populaires

    # On va créer un dataframe avec les champions et leur nombre d'apparition dans les matchs
    champ_match = pd.DataFrame(index=champion['id'], columns=['popularite', 'victoires', 'defaites', 'taux victoire', 'taux defaite'])
    for champ in champion['id']:
        champ_match.loc[champ] = sum(matches['bluetop'] == champ) + sum(matches['bluejungle'] == champ) + sum(matches['bluemid'] == champ) + sum(matches['blueadc'] == champ) + sum(matches['bluesupport'] == champ) + sum(matches['redtop'] == champ) + sum(matches['redjungle'] == champ) + sum(matches['redmid'] == champ) + sum(matches['redadc'] == champ) + sum(matches['redsupport'] == champ)

    # On affiche les champions qui ne sont jamais apparus dans les matchs
    champ_unused = champ_match[champ_match['popularite'] == 0]
    # Et on les enlève du dataframe
    champ_match = champ_match[champ_match['popularite'] != 0]

    blue_wins = matches[matches['result'] == 1]
    red_wins = matches[matches['result'] == 0]

    # On va maintenant créer un dataframe avec les champions et leur nombre de victoire
    for champ in champion['id']:
        champ_match.loc[champ, 'victoires'] = sum(blue_wins['bluetop'] == champ) + sum(blue_wins['bluejungle'] == champ) + sum(blue_wins['bluemid'] == champ) + sum(blue_wins['blueadc'] == champ) + sum(blue_wins['bluesupport'] == champ) + sum(red_wins['redtop'] == champ) + sum(red_wins['redjungle'] == champ) + sum(red_wins['redmid'] == champ) + sum(red_wins['redadc'] == champ) + sum(red_wins['redsupport'] == champ)

    # On va maintenant créer un dataframe avec les champions et leur nombre de défaite
    for champ in champion['id']:
        champ_match.loc[champ, 'defaites'] = sum(red_wins['bluetop'] == champ) + sum(red_wins['bluejungle'] == champ) + sum(red_wins['bluemid'] == champ) + sum(red_wins['blueadc'] == champ) + sum(red_wins['bluesupport'] == champ) + sum(blue_wins['redtop'] == champ) + sum(blue_wins['redjungle'] == champ) + sum(blue_wins['redmid'] == champ) + sum(blue_wins['redadc'] == champ) + sum(blue_wins['redsupport'] == champ)

    # On va maintenant créer un dataframe avec les champions et leur nombre de victoire par rapport à leur nombre d'apparition
    for champ in champion['id']:
        if not champ in champ_unused.index:
            champ_match.loc[champ, 'taux victoire'] = champ_match.loc[champ, 'victoires'] / champ_match.loc[champ, 'popularite']

    # On va maintenant créer un dataframe avec les champions et leur nombre de défaite par rapport à leur nombre d'apparition
    for champ in champion['id']:
        if not champ in champ_unused.index:
            champ_match.loc[champ, 'taux defaite'] = champ_match.loc[champ, 'defaites'] / champ_match.loc[champ, 'popularite']

if affichages['champions_match']: # Affichage des statistiques champion-match (popularité, victoires, défaites)
    if affichages['champions_match_non_used']:
        # On renvoie les champions qui ne sont jamais apparus dans les matchs
        print(champ_unused)

    if affichages['champions_match_victoire']:
        # On renvoie les champions les plus efficaces
        print(champ_match.sort_values(by='taux victoire', ascending=False).head(10))

    if affichages['champions_match_defaite']:
        # On renvoie les champions les moins efficaces
        print(champ_match.sort_values(by='taux defaite', ascending=False).head(10))

    if affichages['champions_match_popularite']:
        # On affiche tout ça
        # plt.figure("Champions les plus populaires")
        champ_match.sort_values(by='popularite', ascending=False)[["popularite", "victoires", "defaites"]].head(10).plot(kind='bar')
        plt.title('Champions les plus populaires')
        plt.xlabel('Champion')
        plt.ylabel('Nombre d\'apparition')
        plt.tight_layout()

    if affichages['champions_match_taux_victoire']:
        # plt.figure("Champions les plus efficaces")
        champ_match.sort_values(by='taux victoire', ascending=False)[["popularite", "victoires", "defaites"]].head(10).plot(kind='bar')
        plt.title('Champions les plus efficaces')
        plt.xlabel('Champion')
        plt.ylabel('Taux de victoire')
        plt.tight_layout()

    if affichages['champions_match_taux_defaite']:
        # plt.figure("Champions les moins efficaces")
        champ_match.sort_values(by='taux defaite', ascending=False)[["popularite", "victoires", "defaites"]].head(10).plot(kind='bar')
        plt.title('Champions les moins efficaces')
        plt.xlabel('Champion')
        plt.ylabel('Taux de défaite')
        plt.tight_layout()

def plot_popularity(nom_roles):
    """
    Affiche les statistiques de popularité des champions pour un rôle donné
    Le rôle doit être écrit sous la forme du suffixe ('top', 'jungle', 'mid', 'adc' ou 'support')
    """
    # On va maintenant créer un dataframe avec les champions et leur nombre de victoire
    for champ in champion['id']:
        champ_match.loc[champ, 'victoires'] = sum(blue_wins['blue' + nom_roles] == champ) + sum(red_wins['red' + nom_roles] == champ)

    # On va maintenant créer un dataframe avec les champions et leur nombre de défaite
    for champ in champion['id']:
        champ_match.loc[champ, 'defaites'] = sum(red_wins['blue' + nom_roles] == champ) + sum(blue_wins['red' + nom_roles] == champ) 

    # On va maintenant créer un dataframe avec les champions et leur nombre de victoire par rapport à leur nombre d'apparition
    for champ in champion['id']:
        if not champ in champ_unused.index:
            champ_match.loc[champ, 'taux victoire'] = champ_match.loc[champ, 'victoires'] / champ_match.loc[champ, 'popularite']

    # On va maintenant créer un dataframe avec les champions et leur nombre de défaite par rapport à leur nombre d'apparition
    for champ in champion['id']:
        if not champ in champ_unused.index:
            champ_match.loc[champ, 'taux defaite'] = champ_match.loc[champ, 'defaites'] / champ_match.loc[champ, 'popularite']

    
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    # On affiche tout ça
    champ_match.sort_values(by='popularite', ascending=False)[["popularite", "victoires", "defaites"]].head(10).plot(ax = ax1, kind='bar')
    ax1.set_title('Champions les plus populaires')
    ax1.set_xlabel('Champion')
    ax1.set_ylabel('Nombre d\'apparition')
    ax1.tick_params(axis='x', labelrotation=0)
    # ax1.set_xticks(rotation=0)
    ax1.autoscale(tight=True)

    champ_match.sort_values(by='taux victoire', ascending=False)[["popularite", "victoires", "defaites"]].head(10).plot(ax = ax2, kind='bar')
    ax2.set_title('Champions les plus efficaces')
    ax2.set_xlabel('Champion')
    ax2.set_ylabel('Taux de victoire')
    ax2.tick_params(axis='x', labelrotation=0)
    ax2.autoscale(tight=True)

    champ_match.sort_values(by='taux defaite', ascending=False)[["popularite", "victoires", "defaites"]].head(10).plot(ax = ax3, kind='bar')
    ax3.set_title('Champions les moins efficaces')
    ax3.set_xlabel('Champion')
    ax3.set_ylabel('Taux de défaite')
    ax3.tick_params(axis='x', labelrotation=0)
    ax3.autoscale(tight=True)

    fig.suptitle('Popularité des champions dans le rôle de \'' + nom_roles + '\'')
    fig.tight_layout()

if affichages['champions_match_par_roles']: # Affichage des statistiques champion-match par rôle (popularité, victoires, défaites)
    if affichages['champions_match_par_roles_top']: # Dans le rôle de top
        plot_popularity('top')
    if affichages['champions_match_par_roles_jungle']: # Dans le rôle de jungle
        plot_popularity('jungle')
    if affichages['champions_match_par_roles_mid']: # Dans le rôle de mid
        plot_popularity('mid')
    if affichages['champions_match_par_roles_adc']: # Dans le rôle de adc
        plot_popularity('adc')
    if affichages['champions_match_par_roles_support']: # Dans le rôle de support
        plot_popularity('support')


if True: # plt.show()
    plt.show()