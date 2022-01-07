import gym
import math
import random
import numpy as np
import argparse


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser(
    description='Run DQN on the game Lunar Lander.')
parser.add_argument('--n-episodes', default=1000, type=int,
                    help='Number of episodes to run.')
parser.add_argument('--max-steps', default=1000, type=int,
                    help='Maximum number of steps for each episode.')
parser.add_argument('--epsilon-start', default=1.,
                    type=float, help='Starting value of epsilon.')
parser.add_argument('--epsilon-end', default=0.01,
                    type=float, help='Final value of epsilon.')
parser.add_argument('--epsilon-decay', default=0.99,
                    type=float, help='Epsilon decay rate.')
parser.add_argument('--update-freq', default=4, type=int,
                    help='How often to update target network.')
parser.add_argument('--gamma', default=0.999,
                    type=float, help='Discount factor.')
parser.add_argument('--seed', default=0, type=int,
                    help='Set seed for the environment.')
parser.add_argument('--lr', default=0.0005, type=float,
                    help='Learning rate for the optimizer.')
parser.add_argument('--tau', default=0.001, type=float,
                    help='For soft update.')
parser.add_argument('--batch-size', default=64, type=int,
                    help='For sampling from replay memory buffer.')
parser.add_argument('--buffer-size', default=100000,
                    type=int, help='Size of memory buffer.')
parser.add_argument('--test', dest='test',
                    action='store_true', help='Test our model.')
parser.add_argument('--render', dest='render', action='store_true',
                    help='Whether to render the environment during testing.')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.seed = random.seed(args.seed)

    def save(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Deep Q Network."""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(args.seed)
        # self.conv1 = nn.Conv2d(in_channels=num_states,
        #                        out_channels=32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(
        #     in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.seed = random.seed(args.seed)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)

        # self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        # or self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(args.buffer_size)
        self.steps = 0

    def act(self, state, epsilon=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_net.eval()
        self.policy_net.train()

        # epsilon-greedy
        if random.random() > epsilon:
            with torch.no_grad():
                # return self.policy_net(state).max(1)[1].view(1, 1)
                return np.argmax(self.policy_net(state).cpu().data.numpy())
        else:
            return self.env.action_space.sample()

    def learn(self, transitions, gamma):
        # unpack transition to state, action, reward, next state

        # index of maximum value for next state

        batch = Transition(*zip(*transitions))
        state_batch = torch.from_numpy(
            np.vstack([x for x in batch.state if x is not None])).float().to(device)
        action_batch = torch.from_numpy(
            np.vstack([x for x in batch.action if x is not None])).long().to(device)
        reward_batch = torch.from_numpy(
            np.vstack([x for x in batch.reward if x is not None])).float().to(device)
        next_state_batch = torch.from_numpy(
            np.vstack([x for x in batch.next_state if x is not None])).float().to(device)
        done_batch = torch.from_numpy(np.vstack(
            [x for x in batch.done if x is not None]).astype(np.uint8)).float().to(device)

        Q_argmax = self.policy_net(next_state_batch).detach()
        _, a_max = Q_argmax.max(1)

        Q_target_next = self.target_net(
            next_state_batch).detach().gather(1, a_max.unsqueeze(1))

        Q_target = reward_batch + \
            (args.gamma * Q_target_next * (1 - done_batch))

        Q_expected = self.policy_net(state_batch).gather(1, action_batch)

        # loss = F.mse_loss(Q_expected, Q_target)

        # or use Huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                args.tau * policy_param.data + (1. - args.tau) * target_param.data)

    def train(self):
        reward_history = []
        rolling_reward_history = deque(maxlen=100)
        epsilon = args.epsilon_start
        for i_episode in range(args.n_episodes):
            state = self.env.reset()
            reward = 0
            for t in range(args.max_steps):
                action = self.act(state, epsilon)
                next_state, r, done, _ = self.env.step(action)
                self.memory.save(state, action, next_state, r, done)

                self.steps = (self.steps + 1) % args.update_freq
                if self.steps == 0:
                    if len(self.memory) >= args.batch_size:
                        transitions = self.memory.sample(args.batch_size)
                        self.learn(transitions, args.gamma)
                state = next_state
                reward += r
                if done:
                    break
            reward_history.append(reward)
            rolling_reward_history.append(reward)
            epsilon = max(args.epsilon_decay * epsilon, args.epsilon_end)

            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(
                i_episode + 1, np.mean(rolling_reward_history)), end="")
            if (i_episode + 1) % 100 == 0:
                print('\rEpisode {}\tAverage Reward: {:.2f}'.format(
                    i_episode + 1, np.mean(rolling_reward_history)))
                torch.save(self.policy_net.state_dict(), 'model.pth'.format(i))
            if np.mean(rolling_reward_history) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode + 1 - 100, np.mean(rolling_reward_history)))
                torch.save(self.policy_net.state_dict(), 'model.pth'.format(i))
                break

        print(reward_history)
        with plt.style.context('seaborn-white'):
            plt.plot(np.arange(len(reward_history)), reward_history)
            plt.xlabel('Episode #')
            plt.ylabel('Reward')
            plt.savefig('dqn-agent-reward{}.pdf'.format(i),
                        bbox_inches='tight')
            plt.gcf().clear()

    def test(self):
        self.policy_net.load_state_dict(torch.load(
            'model.pth', map_location=lambda storage, loc: storage))

        test_scores = []
        for j in range(100):
            state = self.env.reset()
            reward = 0
            for k in range(500):
                action = self.act(state)
                state, r, done, _ = self.env.step(action)
                reward += r
                if args.render:
                    self.env.render()
                if done:
                    print('Episode {}: {}'.format(j + 1, reward))
                    test_scores.append(reward)
                    break

        avg_score = sum(test_scores) / len(test_scores)

        print('\rAverage reward: {:.2f}'.format(avg_score))

if __name__ == '__main__':
    print(args)
    agent = Agent('LunarLander-v2')
    if not args.test:
        agent.train()
    else:
        agent.test()
