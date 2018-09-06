from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K

'''
Reference
=========
Continuous Control With Deep Reinforcement Learning : https://arxiv.org/pdf/1509.02971.pdf
Deep Reinforcement Learning : https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296 

'''


class Critic:
    '''critic value model'''

    def __init__(self, state_size, action_size):
        '''initialize parameters and build model

        Params 
        ======
            state_size (int)    : dimension of each state
            action_size (int)   : dimension of each action

        '''

        self.state_size = state_size
        self.action_size = action_size

        # initialize variables
        self.build_model()

    def build_model(self):
        '''build a critic value network that maps (state, action) pairs -> Q-values'''

        # define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')


        '''try different layer size, regluarization, batch normalization, activation

        Reference: Continuous Control With Deep Reinforcement Learning(2016) 
        === 
        kernel (weight) regularization  : L2 weight decay of 10^-2
        activation      : rectified non-linearity for all hidden layers
        (old) hidden layer    : 2 hidden layers with 400 and 300 units respectively from paper
        (new) hidden layer: 2 hidden layers with size 64, 128 respectively
        output layer    : final output weights were initialized from uniform distribution of (-3e-3, 3e-3)
                          and bias were initialized from uniform distribution of (3e-4, 3e-4)
        learning rate   : 0.001

        '''



        # add hidden layers for states pathway
        net_states = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.01))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=128, kernel_regularizer=regularizers.l2(0.01))(net_states)
        net_states = layers.Activation('relu')(net_states)
        
        # add hidden layers for actions pathway
        net_actions = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.01))(actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dense(units=128, kernel_regularizer=regularizers.l2(0.01))(net_actions)
        net_actions = layers.Activation('relu')(net_actions)

        # combine state and action paths
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # add final output layer to produce action values (Q-values)
        Q_values = layers.Dense(units=1, name='q_values',\
                                kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                bias_initializer=initializers.RandomUniform(minval=-3e-4, maxval=3e-4))(net)

        # create keras model (to simplify code script via a model object)
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')
        
        # Value Loss: L=∑(R_t+1 + Q_t+1 — Qt)²

        # compute action gradients (derivative of Q-values with respect to actions)
        action_gradients = K.gradients(Q_values, actions)

        # define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],\
                                               outputs=action_gradients)

        # The learning phase flag is a bool tensor (0 = test, 1 = train) 
        # to be passed as input to any Keras function 
        # that uses a different behavior at train time and test time.



