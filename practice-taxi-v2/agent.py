import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.05
        self.gamma = 1

    def epsilon_greedy_policy(self, i_episode, state):
        epsilon = 1.0/i_episode

        policy = np.ones(self.nA)*epsilon/self.nA
        policy[np.argmax(self.Q[state])] = 1 - epsilon + epsilon/self.nA

        return policy

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = self.epsilon_greedy_policy(i_episode, state)
        action = np.random.choice(np.arange(self.nA), size=1, p=policy)
        return action[0]

    def update_Q(self, state, action, reward, next_state):
            """ Update state-action value via Q-learning"""
            return self.Q[state][action] + \
                    self.alpha*(reward + self.gamma*(np.max(self.Q[next_state])) - self.Q[state][action])

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.update_Q(state, action, reward, next_state)