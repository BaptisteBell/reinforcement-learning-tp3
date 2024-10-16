"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent

#Suppress Overwriting existing videos warning
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

n_train_epochs = 550

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore

env = gym.wrappers.RecordVideo(env, video_folder="./videos/QLearning", episode_trigger=lambda x: x == 0 or x == n_train_epochs // 2 or x == n_train_epochs - 1)

#################################################
# 1. Play with QLearningAgent
#################################################


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)
        # BEGIN SOLUTION

        agent.update(s, a, r, next_s)
        agent.get_qvalue(s, a)

        total_reward += r 
        s = next_s

        if done:
            break
        # END SOLUTION

    return total_reward


def search_best_parameters(
        env: gym.Env,
        learning_rates: t.List[float],
        epsilons: t.List[float],
        gammas: t.List[float]
    ) -> t.Tuple[t.Dict[str, t.Any], float]:
    '''
    This function searches for the best set of hyperparameters for a Q-Learning agent
    by iterating over different combinations of learning rates, epsilon values, and gamma values.

    Parameters:
    -----------
    env : gym.Env
        The environment in which the agent is trained (e.g., Taxi-v3 from OpenAI Gym).
    
    learning_rates : list of float
        A list of possible learning rates to search over.
    
    epsilons : list of float
        A list of possible epsilon values to search over.
    '''
    
    best_params = dict(
        learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
    )

    best_params_reward = np.NINF

    for lr in learning_rates:
        for e in epsilons:
            for g in gammas:
                agent = QLearningAgent(learning_rate=lr, epsilon=e, gamma=g, legal_actions=list(range(n_actions)))
                rewards = []
                print(f'learning_rate: {lr}, epsilon: {e}, gamma: {g} ...')
                for i in range(1000):
                    rewards.append(play_and_train(env, agent))

                if np.mean(rewards[:100]) > best_params_reward:
                    best_params = dict(
                        learning_rate=lr, epsilon=e, gamma=g, legal_actions=list(range(n_actions))
                    )
                    best_params_reward = np.mean(rewards[:100])
                print(f'rewards is : {np.mean(rewards[:100])}')

    print('------')
    print(f'Best reward is : {best_params_reward}')
    print(f'parameters are -> lr={best_params["learning_rate"]}, epsilon={best_params["epsilon"]}, gamma={best_params["gamma"]}')
    print('------')

    return best_params, best_params_reward

# TODO: créer des vidéos de l'agent en action
agent = QLearningAgent(learning_rate=0.75, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions)))

rewards_QLearning = []
for i in range(n_train_epochs):
    rewards_QLearning.append(play_and_train(env, agent))

env.close()

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

# TODO: créer des vidéos de l'agent en action

agent = QLearningAgentEpsScheduling(
    learning_rate=0.75, epsilon=1, gamma=0.99, legal_actions=list(range(n_actions))
)

env2 = gym.make("Taxi-v3", render_mode="rgb_array")
env2 = gym.wrappers.RecordVideo(env2, video_folder="./videos/QLearningEpsilon", episode_trigger=lambda x: x == 0 or x == n_train_epochs // 2 or x == n_train_epochs - 1)

rewards_QLearningEpsilon = []
for i in range(n_train_epochs):
    rewards_QLearningEpsilon.append(play_and_train(env2, agent))

env2.close()


####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.8, gamma=0.99, legal_actions=list(range(n_actions)))

env3 = gym.make("Taxi-v3", render_mode="rgb_array")
env3 = gym.wrappers.RecordVideo(env3, video_folder="./videos/Sarsa", episode_trigger=lambda x: x == 0 or x == n_train_epochs // 2 or x == n_train_epochs - 1)

rewards_Sarsa = []
for i in range(n_train_epochs):
    rewards_Sarsa.append(play_and_train(env3, agent))

env3.close()

#################################################
# 4. Comparaison des algorithmes
#################################################

def get_first_positive_index(arr):
    for index, value in enumerate(arr):
        if value > 0:
            return index
    return None

columns = ['Max Value', 'MaxValue Epoch', 'Mean Last 100', 'First positive rewards (epoch)']
df = pd.DataFrame(columns=columns)

# QLearning
df.loc['QLearning'] = [np.max(rewards_QLearning), np.argmax(rewards_QLearning), np.mean(rewards_QLearning[-100:]), get_first_positive_index(rewards_QLearning)]
df.loc['QLearningEpsilon'] = [np.max(rewards_QLearningEpsilon), np.argmax(rewards_QLearningEpsilon), np.mean(rewards_QLearningEpsilon[-100:]), get_first_positive_index(rewards_QLearningEpsilon)]
df.loc['Sarsa'] = [np.max(rewards_Sarsa), np.argmax(rewards_Sarsa), np.mean(rewards_Sarsa[-100:]), get_first_positive_index(rewards_Sarsa)]

print(df)
