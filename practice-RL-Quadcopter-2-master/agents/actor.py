from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K

'''
Reference
=========
Continuous Control With Deep Reinforcement Learning : https://arxiv.org/pdf/1509.02971.pdf
Deep Reinforcement Learning : https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296 

'''

class Actor:
    '''actor policy model'''
    
    def __init__(self, state_size, action_size, action_low, action_high):
        '''initialize parameters and build model

        
        Params
        ======
        state_size (int)    : dimension of each state
        action_size (int)   : dimension of each action
        action_low (array)  : min value of each action dimension
        action_high (array) : max value of each action dimension
        
        '''
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # initialize variables
        self.build_model()

    def build_model(self):
        '''build an actor policy network that maps state -> action'''

        # define input layers
        states = layers.Input(shape=(self.state_size,), name='states')

        '''try different layer size, regluarization, batch normalization, activation

        Reference: Continuous Control With Deep Reinforcement Learning(2016) 
        === 
        kernel (weight) regularization  : L2 weight decay of 10^-2
        activation      : rectified non-linearity for all hidden layers
        (old) hidden layer    : 2 hidden layers with 400 and 300 units respectively from paper 
        (new) hidden layer: 3 hidden layers with size 64, 128, 64 respectively
        output layer    : final output weights were initialized from uniform distribution of (-3e-3, 3e-3) 
                          and bias were initialized from uniform distribution of (3e-4, 3e-4)
        learning rate   : 0.001

        '''

        # add hidden layers
        net = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.01))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=128, kernel_regularizer=regularizers.l2(0.01))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.01))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',\
                                   name='raw_actions',\
                                   kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                   bias_initializer=initializers.RandomUniform(minval=-3e-4, maxval=3e-4))(net)
        
        # scale [0,1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x*self.action_range) + self.action_low,\
                                name='actions')(raw_actions)

        # create keras model (to simplify code script via a model object)
        self.model = models.Model(inputs=states, outputs=actions)

        # define loss function using action value (Q-value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients*actions)
        
        # Policy Loss: L = (1/N)*log(𝝅(s)) * Q(s)

        # define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()],\
                                   outputs=[],\
                                   updates=updates_op)
        
        # The learning phase flag is a bool tensor (0 = test, 1 = train) 
        # to be passed as input to any Keras function 
        # that uses a different behavior at train time and test time.

