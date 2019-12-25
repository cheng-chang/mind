import threading
import math

from tqdm import tqdm
import gym
import tensorflow as tf
import numpy as np


EPOCHS = int(1e6)
EPOCH_STEPS = 100
ACTORS = 10
BATCH_SIZE = 64
ACTIONS = 2
ROOT_PRIOR_NOISE_DIRICHLET_ALPHA = 0.03
ROOT_PRIOR_NOISE_FRACTION = 0.25


class RepresentationNet:
  pass


class DynamicsNet:
  pass


class PredictionNet:
  pass


class MuZeroNet:
  def train(self, trajectories):
    pass


class Planner:
  def plan(self, state):
    """Plan the possible future trajectories from raw state and return the action to take."""
    raise NotImplementedError()


class ActionStats:
  def __init__(self, prior):
    self._visit_count = 0
    self._q_value_sum = 0
    self._prior = prior

  def update(self, q_value):
    self._visit_count += 1
    self._q_value_sum += q_value


class Node:
  def __init__(self, parent, state, reward, policy, q_value):
    self._parent = parent
    self._state = state
    self._reward = reward
    prior = self._compute_action_prior(policy, parent is None)
    self._action_stats = {a: ActionStats(prior[a]) for a in ACTIONS}
    self._children = {}

  @staticmethod
  def _compute_action_prior(policy, should_add_noise):
    policy = {a: math.exp(policy[a]) for a in ACTIONS}
    policy_sum = sum(policy.values())
    prior = {a: policy[a] / policy_sum for a in ACTIONS}
    if should_add_noise:
      noise = np.random.dirichlet([ROOT_PRIOR_NOISE_DIRICHLET_ALPHA] * ACTIONS)
      frac = ROOT_PRIOR_NOISE_FRACTION
      prior = {a: (frac * noise[a] + (1 - frac) * prior[a]) for a in ACTIONS}
    return prior

  def is_leaf(self):
    return len(self._children) == 0


class MonteCarloTreeSearch(Planner):
  def plan(self, state):
    pass


class Trajectory:
  """A trajectory is (x0, a0, r0, x1, a1, r1, ..., xn).

  x is the raw observation from the environment,
  a is action,
  r is the immediate reward of the previous action.

  NOTE: x will be transformed to a state representation in the hidden space,
  the transformed state uses symbol s.
  """
  def __init__(self, x0):
    self._states = [x0]
    self._actions = []
    self._rewards = []

  def current_state(self):
    return self._states[-1]

  def append(self, current_action, reward, next_state):
    self._actions.append(current_action)
    self._rewards.append(reward)
    self._states.append(next_state)


class Memory:
  def store(self, trajectory):
    pass

  def size(self):
    pass

  def sample(self, batch_size):
    assert(self.size() >= batch_size)
    pass


class Actors:
  def run(self):
    raise NotImplementedError()


class ThreadActors(Actors):
  def __init__(self, num_actors, fn, args):
    self._threads = []
    for _ in range(num_actors):
      self._threads.append(threading.Thread(target=fn, args=args))

  def run(self):
    for t in self._threads:
      t.start()
    for t in self._threads:
      t.join()


def play(env, planner, memory):
  """Reset the environment and collect a new trajectory into memory."""
  trajectory = Trajectory(env.reset())
  for _ in tqdm(range(EPOCH_STEPS)):
    x = trajectory.current_state()
    a = planner.plan(x)
    nx, r, done, _ = env.step(a)
    trajectory.append(a, r, nx)
    if done:
      break
  memory.store(trajectory)


def train(env, muzero, planner, memory):
  actors = ThreadActors(ACTORS, play, (env, planner, memory))
  actors.run()
  if memory.size() >= BATCH_SIZE:
    trajectories = memory.sample(BATCH_SIZE)
    muzero.train(trajectories)


if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  muzero = MuZeroNet()
  memory = Memory()
  planner = MonteCarloTreeSearch()
  try:
    for epoch in range(EPOCHS):
      print('Epoch ' + epoch)
      train(env, muzero, planner, memory)
  finally:
    env.close()
