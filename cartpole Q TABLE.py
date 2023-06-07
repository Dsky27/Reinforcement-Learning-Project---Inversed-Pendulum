
from sklearn.preprocessing import KBinsDiscretizer
import gym
import numpy as np
import random as rd, time, math

'''
Cart pole implementation with Q-learning (no DQN)


'''

#create the environment
env = gym.make('CartPole-v1', render_mode='human')
obs = env.reset()
env.render()


# We need a discrete environment to implement Q-learning: discretization constants:
env_dsize = (6, 12)
lower_bound = [env.observation_space.low[2], -math.radians(50)]
upper_bound = [env.observation_space.high[2], math.radians(50)]

# Q learning constants
Q_table = np.zeros(env_dsize + (env.action_space.n,))
discount_factor = 0.9
learning_rate = 0.7
epsilon, epmax, epmin, epdecay = 1, 1, 0.05, 0.01
N_episodes = 10000


# exonential epsilon decay 
def update_epsilon(N):
    global epsilon
    epsilon = epmin + (epmax - epmin) * math.exp(-epdecay * N)

# discretizing space of observations (were not using the position and the velcity of the cart here )
def discretizer(_, __, angle, angleV):
    est = KBinsDiscretizer(n_bins=env_dsize, encode='ordinal', strategy='uniform')
    est.fit([lower_bound, upper_bound])
    return tuple(map(int, est.transform([[angle, angleV]])[0]))

# getting the max Q value in the table to take an action
def policy(state : tuple):
    return np.argmax(Q_table[state])

# belleman equation
def new_Q_value( reward : float ,  new_state : tuple ) -> float:
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


# training loop
for e in range(N_episodes):
    #resetting the state and the buffer variables for episode end, updating epsilon
    # d1 : terminated : if the cart goes to far one direction or if the pole falls 
    # d2 : truncated : if the episode reaches its maximum duration (500 for CartPole-v1)
    current_state, d1, d2 = discretizer(*env.reset()[0]), False, False
    update_epsilon(e)

    print(e, "  ", epsilon)

    # episode conduction
    while d1 == False and d2 == False:
        # choosing an action between exploration and exploitation
        action = policy(current_state)
        if rd.random() < epsilon:
            action = rd.randint(0,1)


        obs, reward, d1, d2, _ = env.step(action)
        new_state = discretizer(*obs)

        #updating the q values
        learnt_Q_value = new_Q_value(reward, new_state)
        old_Q_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1-learning_rate) * old_Q_value + learning_rate * learnt_Q_value

        current_state = new_state

        #displaying the environment
        env.render()


