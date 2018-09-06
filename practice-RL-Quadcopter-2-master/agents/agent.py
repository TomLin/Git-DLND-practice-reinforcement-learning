from agents.actor import Actor
from agents.critic import Critic
from agents.OUNoise import OUNoise
from agents.replay_buffer import ReplayBuffer
import numpy as np

class DDPG():
    '''reinforcement learning agent that learns using Deep Deterministic Policy Gradient'''
    def __init__(self, task):
        '''
        Params
        ======
        task (object)   : environment

        '''

        '''
        Reference: Continuous Control With Deep Reinforcement Learning(2016)
        Playing CartPole through Asynchronous Advantage Actor Critic (A3C) with tf.keras
        =========
        gamma   : 0.99
        tau     : 0.001
        buffer_size (ReplayBuffer)  : 1e6
        batch_size (ReplayBuffer)   : 64
        theta (Ornstein-Uhlenbeck process)  : 0.15
        sigma (Ornstein-Uhlenbeck process)  : 0.2


        '''

        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # actor (policy) model - use two copies of model for updating model and producing target 
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # critic (value) model - use two copies of model for updating model and producing target
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # initialize target model parameters with local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, 
                             self.exploration_theta, self.exploration_sigma)
        
        # replay memory
        self.buffer_size = 1000000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # algorithm parameters
        self.gamma = 0.99 # discount factor
        self.tau = 0.001 # for soft update of target parameters

        # reward history
        self.best_avg_score = -np.inf
        self.accumulated_reward = 0
        self.count = 0

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.accumulated_reward = 0
        self.count = 0
        return state
    
    def step(self, action, reward, next_state, done):
        # save experience and reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # roll over last state and action
        self.last_state = next_state
        
        # accumulate reward
        self.accumulated_reward += reward
        self.count += 1

        # record best average score
        if done:
            if float(self.accumulated_reward/self.count) > self.best_avg_score:
                self.best_avg_score = float(self.accumulated_reward/self.count)

    def act(self, state):
        '''returns actions for given state(s) as per current policy'''
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample()) # add some noise for exploration
        # both action and self.noise.sample() are numpy object, + means sum up both,
        # instead of concatenation

    def learn(self, experiences):
        '''update policy and value parameters using given batch of experience tuples'''
        # convert experience tuples to separate arrays for each element(states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).\
                           astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).\
                           astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).\
                           astype(np.uint8).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # get predicted next-state actions and Q-values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # compute Q targets for current states and train critic model (local)
        # Value Loss: L=∑(R_t+1 + Q_t+1 — Qt)²
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # train actor model (local)
        # Policy Loss: L = (1/N)*log(𝝅(s)) * Q(s)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                      (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1]) # custom training function
        # The learning phase flag is a bool tensor (0 = test, 1 = train)
        # to be passed as input to any Keras function
        # that uses a different behavior at train time and test time.

        # soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        '''soft update model parameters'''
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights),\
            'Local and target model parameters must have the same size'
        
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        




    