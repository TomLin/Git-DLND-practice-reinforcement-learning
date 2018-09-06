import random
from collections import namedtuple, deque

class ReplayBuffer:
    '''fixed-size buffer to store experience tuples'''

    def __init__(self, buffer_size, batch_size):
        '''initialize a ReplayBuffer object

        Params
        ======
        buffer_size (int)   : maximum size of buffer
        batch_size (int)    : size of each training batch

        '''

        '''
        Reference: Continuous Control With Deep Reinforcement Learning(2016)
        =========
        buffer_size : 1e6 
        batch_size  : 64

        '''
        
        self.memory = deque(maxlen=buffer_size) # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state','action','reward','next_state','done'])

    def add(self, state, action, reward, next_state, done):
        '''add a new experience to memory'''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        '''randomly sample a batch of experiences from memory'''

        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        '''return the current size of internal memory'''
        
        return len(self.memory)


