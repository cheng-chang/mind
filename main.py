import random

import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


EPOCHS = 10000
EPOCH_STEPS = 1000
EPSILON = 0.01
GAMMA = 0.01
PREPROCESSED_IMAGE_SHAPE = (110, 84)
DQN_INPUT_SHAPE = (*PREPROCESSED_IMAGE_SHAPE, 4)
MEMORY_SIZE = 100
SAMPLE_SIZE = 100
ACTIONS = 4


def random_action():
  return int(random.random() * ACTIONS)


def preprocess_raw_image(image):
  """Preprocess raw image returned from simulator.

  1. RGB to grayscale
  2. downsampling to INPUT_SHAPE

  Arguments:
    image: (210, 160, 3) ndarray
  """
  gray = tf.image.rgb_to_grayscale(image)
  resized = tf.image.resize(gray, PREPROCESSED_IMAGE_SHAPE)
  return resized


class DQN:
  def __init__(self):
    self.Q = _build_dqn()

  def _build_dqn():
    """Build the Deep-Q-Network."""
    Q = keras.Sequential()
    Q.add(layers.Conv2D(16, (8, 8), 4, activation='relu', input_shape=DQN_INPUT_SHAPE))
    Q.add(layers.Conv2D(32, (4, 4), 2, activation='relu'))
    Q.add(layers.Flatten())
    Q.add(layers.Dense(256, activation='relu'))
    Q.add(layers.Dense(ACTIONS, activation='softmax'))
    return Q

  def action(self, state):
    pass

  def _max_q_value(self, state):
    pass

  def optimize(self, transitions):
    pass


class Memory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []

  def add(self, s, a, r, ns):
    pass

  def sample(self, size):
    pass

  def _evict(self):
    pass


class Trajectory:
  def __init__(self, initial_state):
    pass

  def add(self, a, x):
    pass

  def state(self):
    pass


def train():
  env = gym.make('Breakout-v0')
  Q = DQN()
  memory = Memory(MEMORY_SIZE)
  for epoch in range(EPOCHS):
    print('Epoch ', epoch)
    rewards = 0
    steps = 0
    traj = Trajectory(env.reset())
    for t in tqdm(range(EPOCH_STEPS)):
      s = traj.state()
      if random.random() <= EPSILON:
        a = random_action()
      else:
        a = Q.action(s)
      x, r, done, _ = env.step(a)
      rewards += r
      steps += 1
      traj.add(a, x)
      memory.add(s, a, r, traj.state())
      Q.optmize(memory.sample(SAMPLE_SIZE))
      if done:
        break
    print('steps = {}, rewards = {}'.format(steps, rewards))
  env.close()


if __name__ == '__main__':
  train()
