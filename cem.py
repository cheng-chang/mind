from __future__ import print_function

import sys
import gym
from gym import wrappers, logger
import numpy as np
from six.moves import cPickle as pickle
import json, sys, os
from os import path
import argparse
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]
    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a


class ContinuousActionLinearPolicy(object):
    def __init__(self, theta, n_in, n_out):
        assert len(theta) == (n_in + 1) * n_out
        self.W = theta[0 : n_in * n_out].reshape(n_in, n_out)
        self.b = theta[n_in * n_out : None].reshape(1, n_out)
    def act(self, ob):
        ob = ob.reshape((1, 3))
        a = ob.dot(self.W) + self.b
        if a[0][0] < -2:
            return np.array([-2])
        if a[0][0] > 2:
            return np.array([2])
        return a.reshape((1,))



class ContinuousNNPolicy(nn.Module):
    HIDDEN_SIZE = 16

    @classmethod
    def theta_size(cls, env):
        s_size = env.observation_space.shape[0]
        h_size = cls.HIDDEN_SIZE
        a_size = env.action_space.shape[0]
        return (s_size + 1) * h_size + (h_size + 1) * a_size

    def __init__(self, theta, env):
        super(ContinuousNNPolicy, self).__init__()
        self.hidden_size = self.HIDDEN_SIZE
        self.observation_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.fc1 = nn.Linear(self.observation_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)
        self._set_params(theta)

    def to_tensor(self, x):
        return torch.from_numpy(x).float()

    def _set_params(self, theta):
        s_size = self.observation_size
        h_size = self.hidden_size
        a_size = self.action_size
        fc1_end = (s_size * h_size) + h_size
        fc1_w = self.to_tensor(theta[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = self.to_tensor(theta[s_size*h_size:fc1_end])
        fc2_w_end = fc1_end + (h_size * a_size)
        fc2_w = self.to_tensor(theta[fc1_end:fc2_w_end].reshape(h_size, a_size))
        fc2_b = self.to_tensor(theta[fc2_w_end:])
        self.fc1.weight.data.copy_(fc1_w.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_w.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def act(self, ob):
        x = F.relu(self.fc1(self.to_tensor(ob)))
        x = F.tanh(self.fc2(x))
        return x.detach().numpy()


def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[-n_elite:]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        # NOTE(cc):
        # If th_std keeps constant, it advocates exploration even when theta reaches
        # optimal point. If it changes along with elite_ths,
        # then the std should become smaller and smaller while theta is closer to
        # the optimal point, so less and less exploration will happen, but it could also
        # stuck at local sub-optimal minima.
        # In the experiment of ContinuousMountainCar, keeping th_std constant can solve
        # the problem, while updating th_std can lead to sub-optimal local minima.
        #th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}


def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1


def show_learning_curve(rewards):
    plt.plot(np.arange(1, len(rewards) + 1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('target', nargs="?", default='MountainCarContinuous-v0')
    args = parser.parse_args()

    env = gym.make(args.target)
    env.seed(101)
    np.random.seed(101)
    params = dict(n_iter=1000000, batch_size=50, elite_frac=0.2, initial_std=0.5)
    # NOTE(cc):
    # if num_steps is not large enough,
    # the mountain car has little chance to reach the target,
    # so it has little chance to get the big reward,
    # which causes it to stuck at the learning process without progress.
    # This situation also happens in Montezuma's Revenge.
    num_steps = 5000

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    #outdir = '/tmp/cem-agent-results'
    outdir = tempfile.mkdtemp()
    #env = wrappers.Monitor(env, outdir, force=True)

    # Prepare snapshotting
    # ----------------------------------------
    # def writefile(fname, s):
    #     with open(path.join(outdir, fname), 'w') as fh: fh.write(s)

    def noisy_evaluation(theta):
        #agent = BinaryActionLinearPolicy(theta)
        #agent = ContinuousActionLinearPolicy(theta, 3, 1)
        agent = ContinuousNNPolicy(theta, env)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # Train the agent, and snapshot each stage
    theta0 = params['initial_std'] * np.ones(ContinuousNNPolicy.theta_size(env))
    rewards = []
    for (i, iterdata) in enumerate(cem(noisy_evaluation, theta0, **params)):
        #print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        if i % 10 == 0:
            print('CHECKPOINT')
        theta = iterdata['theta_mean']
        reward = noisy_evaluation(theta)
        mean_episode_reward = iterdata['y_mean']
        print('{},{},{}'.format(i, mean_episode_reward, reward))
        rewards.append(mean_episode_reward)
        # sys.stdout.write("{},{}\n".format(i, iterdata['y_mean']))
        # sys.stdout.flush()
        #agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        #agent = ContinuousActionLinearPolicy(iterdata['theta_mean'], 3, 1)
        #if args.display: do_rollout(agent, env, 200, render=True)
        #writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))
    env.close()

    show_learning_curve(rewards)
