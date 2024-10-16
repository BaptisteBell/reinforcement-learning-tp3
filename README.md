# Reinforcement Learning TP3

Indications dans le fichier taxi.py

## Description

Dans ce projet, nous avons implémenté des agents utilisant les algorithmes Q-Learning et SARSA pour jouer au jeu Taxi-v3 de OpenAI Gymnasium. Le but du jeu est de déplacer un taxi dans une grille 5x5, récupérer un passager et le déposer à une destination spécifique en un minimum d'actions.

Les algorithmes testés sont :
- Q-Learning
- Q-Learning avec epsilon scheduling
- SARSA

Nous avons comparé les performances des algorithmes en fonction de plusieurs critères, tels que la récompense maximale obtenue, le nombre d'épisodes avant d'obtenir une première récompense positive, et la stabilité des récompenses sur les derniers épisodes.

## Installation et Lancement

### Prérequis

Assurez-vous d'avoir Python 3 installé sur votre machine.

### Installation des dépendances

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

### Lancement du TP

nsuite, pour lancer l'algorithme, exécutez le fichier `taxi.py` avec la commande suivante :

```bash
python3 taxi.py
```

## Résulat et Comparaison

La comparaison des algorithmes ainsi que les choix d'implémentation sont détaillés dans le rapport `Rapport_TP3_ReinforcementLearning.pdf`.
