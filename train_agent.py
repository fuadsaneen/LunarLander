# Landing pad is always at coordinates (0,0). 
# Coordinates are the first two numbers in state vector. 
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. 
# If lander moves away from landing pad it loses reward back. 
# Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. 
# Each leg ground contact is +10. 
# Firing main engine is -0.3 points each frame. 
# Solved is 200 points. 
# Landing outside landing pad is possible. 
# Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 
# Action is two real values vector from -1 to +1. 
# First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power. 
# Engine can’t work with less than 50% power. 
# Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.

# Import functions.
import gym
import random
from tensorflow.keras import Sequential
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.activations import relu, linear
import numpy as np
import argparse
import os

# Initialize simulator.
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

# DQN definition.
class DQN:

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        self.model.save("agent.h5", save_format="h5")


# Train DQN.
def train_dqn(episode):

    # Store all loses.
    loss = []

    # Declare agent.
    agent = DQN(env.action_space.n, env.observation_space.shape[0])

    # For each episode.
    for e in range(1, episode + 1):

        # Reset environment.
        state = env.reset()
        state = np.reshape(state, (1, 8))

        # Reset score.
        score = 0

        # Declare maximum steps.
        max_steps = 3000

        # For each step.
        for i in range(max_steps):

            action = agent.act(state)
            # env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

            # Print score.
            if done:
                print("\nEpisode Number: {} \nCurrent Score: {}".format(e, score))
                break

        # Push score to the list.    
        loss.append(score)

        # Average score of last 100 episode.
        is_solved = np.mean(loss[-100:])
        print("Average score over last 100 episodes: {0:.2f}\n".format(is_solved))
        # if is_solved > 200:
        #     print('\nTask Status: Successfully Completed! \n')
        #     break

    # Save agent.
    agent.save()

    # Return loss list.
    return loss

# Main function.
if __name__ == '__main__':

    episodes = 400

    print("\n\nTraining agent till the average score over last 100 episodes becomes 200 and more.\n\n")
    loss = train_dqn(episodes)

    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()