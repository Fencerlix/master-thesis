import gym
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt 

import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

# from IPython.display import clear_output

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY = 100

DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# For stats
STATS_EVERY = 500
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}


q_table = np.random.uniform(low=0, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

# TRAINING (Q-Learning)
for episode in range(EPISODES):
    # set up episode
    # print(f"episode {episode}")
    episode_reward = 0
    discrete_state, _ = env.reset()
    discrete_state = get_discrete_state(discrete_state)
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:

        # epsilon greedy action selection
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        '''
        if episode % SHOW_EVERY == 0:
            env.render()
        '''

        # If simulation did not end yet after last step - update Q table
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) # Maximum possible Q value in next step (for new state)
            current_q = q_table[discrete_state + (action,)] # Current Q value (for current state and performed action)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)


        # Simulation ended - update Q directly
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying (as long as episode is in the decaying range)
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')


env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()