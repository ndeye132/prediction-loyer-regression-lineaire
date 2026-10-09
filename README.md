# Prédiction du loyer par régression linéaire

Projet IA, Licence 2 MPI, Université Amadou Mahtar Mbow (UAM).
Réalisée par Ndeye MBODJI.

## Description

Ce projet prédit le loyer d'un logement à partir de sa surface, de son nombre de pièces, de son quartier et de son étage, avec une régression linéaire multiple (scikit-learn). Les données sont fictives : 120 logements générés dans le code.

## Contenu du dépôt

- `projet_ia.py` : code complet (génération des données, corrélations, modèle, évaluation, graphiques)
- `dataset_loyer.csv` : jeu de données de 120 logements

## Bibliothèques nécessaires

numpy, pandas, matplotlib, seaborn, scikit-learn

    python -m pip install numpy pandas matplotlib seaborn scikit-learn

## Lancer le projet

Depuis le dossier du projet :

    python projet_ia.py

Le script affiche les résultats dans le terminal et enregistre les graphiques au format PNG dans le même dossier.
