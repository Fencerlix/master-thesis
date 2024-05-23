from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from collections import deque
import random
import numpy as np

# TODO: Follow standards everywhere, func names.


# DQN model
class DQN:
    def __init__(self, env):
        # Env for input, output dimensions
        self.env = env

        # Epsilon for exploitation/exploration.
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0.01

        self.learning_rate = 0.001

        # Mini batch size.
        self.batch_size = 25
        # Gamma, or importance of future rewards.
        self.gamma = 0.99
        
        # Replay memory.
        self.replayBuffer = deque(maxlen=20000)

        # Main model
        self.model = self.build_network()

        # Target model for future Q values.
        # More stable approach.
        self.targetModel = self.build_network()
        # Initialize targetModel weights with model weights 
        self.update_weights()

    # Create network
    def build_network(self):
        # Get input shape for network inputs
        input_shape = self.env.observation_space.shape

        mdl = Sequential()
        mdl.add(Dense(24, activation = 'relu', input_shape = input_shape))
        mdl.add(Dense(48, activation = 'relu'))
        mdl.add(Dense(self.env.action_space.n, activation = 'linear'))
        mdl.compile(loss = 'mse', optimizer=adam_v2(lr = self.learning_rate))
        return mdl


    def predict_action(self, state):
        state = state.reshape(1, self.env.observation_space.shape[0])
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(self.model.predict(state)[0])

        return action

    # Save replay to memory
    def save_replay(self, currentState, action, reward, new_state, done):
        new_state = new_state.reshape(1, len(self.env.observation_space.low) )
        self.replayBuffer.append([currentState, action, reward, new_state, done])

    # Sync targetMode land model
    def update_weights(self):
        self.targetModel.set_weights(self.model.get_weights())

    def train(self):
        # If not enough replays yet
        if len(self.replayBuffer) < self.batch_size:
            return

        # Get a random sample
        samples = random.sample(self.replayBuffer,self.batch_size)

        # Extract states and new states into their own arrays
        states = []
        new_states = []
        for sample in samples:
            state, _, _, new_state, _ = sample
            states.append(state)
            new_states.append(new_state)

        # Turn state and new_state arrays into numpy arrays
        # for fitting. This is a lot faster.
        states = np.array(states).reshape(self.batch_size, self.env.observation_space.shape[0])
        new_states = np.array(new_states).reshape(self.batch_size, self.env.observation_space.shape[0])

        # List of current predicts
        targets = self.model.predict(states)
        # List of future predicts
        new_state_targets = self.targetModel.predict(new_states)

        # Index counter
        i=0
        # Loop through every sample
        for sample in samples:
            # Get action, reward and done
            _, action, reward, _, done = sample
            if done:
                targets[i][action] = reward
            else:
                # Future Q-reward. 
                Q_future = max(new_state_targets[i])
                targets[i][action] = reward + Q_future * self.gamma
            i+=1

        self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, name):
        self.model.save_weights('./' + name + '.h5')

    def load(self, name):
        self.model.load_weights('./' + name)
        # Set exploration to min
        self.epsilon = self.epsilon_min
        # Set target model weights to match model
        self.update_weights()