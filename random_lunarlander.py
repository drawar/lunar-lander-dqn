import gym
import math
import random
import numpy as np
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RandomAgent:

    def __init__(self, env_name, max_eps):
        self.env = gym.make(env_name)
        self.max_episodes = max_eps

    def run(self):
        reward_history = []
        steps_history = []
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, r, done, _ = self.env.step(
                    self.env.action_space.sample())
                steps += 1
                reward += r
            reward_history.append(reward)
            steps_history.append(steps)
        final_avg = sum(reward_history) / float(self.max_episodes)
        steps_avg = sum(steps_history) / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(
            self.max_episodes, final_avg))
        print("Average step across {} episodes: {}".format(
            self.max_episodes, steps_avg))

        with plt.style.context('seaborn-white'):
            plt.hist(reward_history, bins=20)
            plt.annotate('Mean reward: {}'.format(
                int(final_avg)), xy=(-600, 200))
            plt.xlabel('Frequency')
            plt.ylabel('Reward')
            plt.savefig('rand-agent-reward.pdf', bbox_inches='tight')
            plt.gcf().clear()

        with plt.style.context('seaborn-white'):
            plt.hist(steps_history, bins=20)
            # plt.annotate('Mean episode length: {}'.format(int(final_avg)), xy=(-600,200))
            plt.xlabel('Frequency')
            plt.ylabel('Episode length')
            plt.savefig('rand-agent-episode-length.pdf', bbox_inches='tight')
            plt.gcf().clear()

if __name__ == '__main__':
    agent = RandomAgent('LunarLander-v2', 1000)
    agent.run()
