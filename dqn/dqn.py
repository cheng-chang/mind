import os
import random
from datetime import datetime

import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


def timestamp():
  return datetime.now().strftime("%Y%m%d-%H%M%S")


EPOCHS = 1000000
EPOCH_STEPS = 10000
EPSILON_MAX = 1
EPSILON_MIN = 0.1
GAMMA = 0.99
MEMORY_SIZE = 100000
SAMPLE_SIZE = 32
# How many most recent consequent raw states to consider as one processed state
HORIZON_SIZE = 4
LOG_DIR="logs/dqn/" + timestamp()
TENSORBOARD_CALLBACK = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, profile_batch = 3)


class GymEnv:
  def __init__(self, name):
    self._name = name
    self._env = gym.make(name)
    self._state_shape = self._env.reset().shape

  def name(self):
    return self._name

  def state_shape(self):
    return self._state_shape

  def action_shape(self):
    return self._env.action_space.n

  def step(self, action):
    return self._env.step(action)

  def reset(self):
    return self._env.reset()

  def random_action(self):
    return self._env.action_space.sample()

  def preprocess_state(self, state):
    """Preprocess the raw state.

    For example, rescale the image, turn the image to grayscale, etc.
    Or nothing.

    Returns:
      numpy ndarray
    """
    raise NotImplementedError


class CartPoleEnv(GymEnv):
  def __init__(self):
    super().__init__('CartPole-v0')

  def preprocess_state(self, state):
    return state


class BreakoutEnv(GymEnv):
  PREPROCESSED_IMAGE_SHAPE = (84, 84)

  def __init__(self):
    super().__init__('Breakout-v0')

  def preprocess_state(self, image):
    """Preprocess raw image returned from simulator.

    RGB to grayscale and downsampling.

    Args:
      image: RAW_IMAGE_SHAPE ndarray

    Returns:
      ndarray of shape PREPROCESSED_IMAGE_SHAPE
    """
    resized = tf.image.resize(image, PREPROCESSED_IMAGE_SHAPE)
    return tf.reshape(tf.image.rgb_to_grayscale(resized), PREPROCESSED_IMAGE_SHAPE).numpy()


class ExplorationRate:
  """When a random value between [0, 1] is <= this ratio, take random actions."""
  def value(self, total_steps):
    raise NotImplementedError


class LinearExplorationRate(ExplorationRate):
  def __init__(self, max_value, min_value, steps):
    """Decay epsilon linearly from max_value to min_value in the specified number of steps."""
    self._max = max_value
    self._min = min_value
    self._decay_per_step = (max_value - min_value) / steps

  def value(self, total_steps):
    return self._max - self._decay_per_step * total_steps


def image_model(input_shape, output_shape):
  """Build the Deep-Q-Network for raw image inputs."""
  Q = keras.Sequential()
  Q.add(layers.Conv2D(16, 8, 4, activation='relu', input_shape=input_shape))
  Q.add(layers.Conv2D(32, 4, 2, activation='relu'))
  Q.add(layers.Flatten())
  Q.add(layers.Dense(256, activation='relu'))
  Q.add(layers.Dense( output_shape))
  return Q


def vector_model(input_shape, output_shape):
  """Build the Deep-Q-Network for vector inputs."""
  return keras.Sequential([
    layers.Flatten(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(output_shape)
  ])


def build_model(env):
  state_shape = env.state_shape()
  input_shape = (*state_shape, HORIZON_SIZE)
  output_shape = env.action_shape()
  if len(state_shape) == 1:
    return vector_model(input_shape, output_shape)
  return image_model(input_shape, output_shape)


class DQN:
  def __init__(self, model):
    self._Q = model
    self._optimizer = keras.optimizers.Adam()

  def _q_values(self, state):
    """Compute a forward pass of state through the Q network.

    Args:
      state is returned by Trajectory.state().

    Returns:
      a Tensor with Q values (scores) for each action, tensor shape is (4, )
    """
    return self._Q(np.array([state]))[0]

  def action(self, state):
    return int(tf.math.argmax(self._q_values(state)))

  def _max_q_value(self, state):
    return float(tf.math.reduce_max(self._q_values(state)))

  def _max_rewards(self, transition):
    s, a, r, ns, done = transition
    if done:
      return r
    return r + GAMMA * self._max_q_value(ns)

  def _loss(self, transition):
    s, a, r, ns, done = transition
    return (self._max_rewards(transition) - self._q_values(s)[a]) ** 2

  def optimize(self, transitions):
    with tf.GradientTape() as g:
      loss = tf.math.reduce_sum([self._loss(t) for t in transitions])
    w = self._Q.trainable_weights
    grads = g.gradient(loss, w)
    self._optimizer.apply_gradients(zip(grads, w))


class Memory:
  """Memory of transitions.

  Each transition is (state, action, reward, next_state, done).
  state is defined by Trajectory.state().

  Has a maximum capacity, when capacity is overlimit,
  eviction happens automatically.
  """
  def __init__(self, capacity):
    self._capacity = capacity
    self._memory = []

  def add(self, s, a, r, ns, done):
    """Add a new transition.

    If the memory is already full, evict one before adding.
    """
    if len(self._memory) == self._capacity:
      self._evict()
    self._memory.append((s, a, r, ns, done))

  def sample(self, size):
    """Randomly sample a batch of transitions from the memory.

    The batch size is min(size, memory size).
    """
    return random.choices(self._memory, k=size)

  def size(self):
    return len(self._memory)

  def _evict(self):
    """Evict the oldest transition."""
    self._memory.pop(0)


class Trajectory:
  """Trajectory of raw state and actions ordered by time.

  x is the raw image frame returned by the emulator.
  a is the action.

  When environment is reset, there is an initial raw image x0,
  so a trajectory is (x0, a0, x1, a1, x2, ... ) until an epoch
  is done or epoch steps are exhausted.

  Args:
    env: the env to be reset
  """
  def __init__(self, env):
    self._env = env
    self._traj = [self._env.reset()]

  def add(self, a, x):
    """Add (a, x) to the trajectory.

    a is the action taken at the current state,
    x is the state transitioned to after a.
    """
    self._traj += [a, x]

  @staticmethod
  def _left_zero_padding(arr, length, shape):
    if len(arr) >= length:
      return arr
    return [np.zeros(shape)] * (length - len(arr)) + arr

  def _horizon(self):
    return self._traj[::2][-HORIZON_SIZE:]

  def state(self):
    """Preprocess the last HORIZON_SIZE raw states to a state.

    The preprocessed states are stacked together in increasing time order.
    If there are fewer than HORIZON_SIZE raw states, left padding with zeros.

    Returns:
      Tensor of shape (*state_shape, HORIZON_SIZE)
    """
    states = self._left_zero_padding(self._horizon(), HORIZON_SIZE, self._env.state_shape())
    return tf.stack([self._env.preprocess_state(s) for s in states], -1)


def train(env):
  Q = DQN(build_model(env))
  memory = Memory(MEMORY_SIZE)
  total_steps = 0
  eps = LinearExplorationRate(EPSILON_MAX, EPSILON_MIN, MEMORY_SIZE)
  for epoch in range(1, EPOCHS + 1):
    print('Epoch {} / {}'.format(epoch, EPOCHS))
    rewards = 0
    steps = 0
    traj = Trajectory(env)
    for step in tqdm(range(EPOCH_STEPS)):
      s = traj.state()
      if random.random() <= eps.value(total_steps):
        a = env.random_action()
      else:
        a = Q.action(s)
      x, r, done, _ = env.step(a)
      steps += 1
      rewards += r
      traj.add(a, x)
      memory.add(s, a, r, traj.state(), done)
      if memory.size() > SAMPLE_SIZE:
        Q.optimize(memory.sample(SAMPLE_SIZE))
      if done:
        break
    total_steps += steps
    print('time = {}, steps = {}, rewards = {}, total steps = {}'.format(timestamp(), steps, rewards, total_steps))
    os.sys.stdout.flush()


if __name__ == '__main__':
  env = CartPoleEnv()
  try:
    train(env)
  finally:
    env.close()
