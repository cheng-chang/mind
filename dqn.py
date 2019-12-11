import random

import numpy as np
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


class DQN:
  def __init__(self):
    self._Q = _build_dqn()

  def _build_dqn():
    """Build the Deep-Q-Network."""
    Q = keras.Sequential()
    Q.add(layers.Conv2D(16, (8, 8), 4, activation='relu', input_shape=DQN_INPUT_SHAPE))
    Q.add(layers.Conv2D(32, (4, 4), 2, activation='relu'))
    Q.add(layers.Flatten())
    Q.add(layers.Dense(256, activation='relu'))
    Q.add(layers.Dense(ACTIONS))
    return Q

  def _random_action(self):
    return int(random.random() * ACTIONS)

  def _optimal_action(self, state):
    scores = self._Q.predict(np.array([state]))[0]
    return int(tf.math.argmax(scores).numpy())

  def action(self, state):
    return self._random_action() if random.random() <= EPSILON else self._optimal_action(state)

  def _max_q_value(self, state):
    pass

  def optimize(self, transitions):
    pass


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
    random.shuffle(self._memory)
    return self._memory[:size]

  def _evict(self):
    """Evict a transition when memory is full.

    If sample is never called, then evicts the oldest transition.
    If sample is ever called, then the first transition is a random one.
    """
    self._memory.pop(0)


class Trajectory:
  """Trajectory of raw state and actions ordered by time.

  x is the raw image frame returned by the emulator.
  a is the action.

  When environment is reset, there is an initial raw image x0,
  so a trajectory is (x0, a0, x1, a1, x2, ... ) until an epoch
  is done or epoch steps are exhausted.
  """
  def __init__(self, x0):
    self._traj = [x0]

  def add(self, a, x):
    """Add (a, x) to the trajectory.

    a is the action taken at the current state,
    x is the state transitioned to after a.
    """
    self._traj += [a, x]

  def _preprocess_raw_image(self, image):
    """Preprocess raw image returned from simulator.

    RGB to grayscale and downsampling.

    Arguments:
      image: (210, 160, 3) ndarray

    Returns:
      tensor of shape PREPROCESSED_IMAGE_SHAPE
    """
    resized = tf.image.resize(image, PREPROCESSED_IMAGE_SHAPE)
    return tf.image.rgb_to_grayscale(resized)

  def state(self):
    """Preprocess the last 4 raw images to a state.

    Each image is processed to a matrix,
    then stack the 4 matrices together in increasing time order.
    return ndarray with shape DQN_INPUT_SHAPE.
    """
    assert(len(self._traj) >= 7)
    last_4_images = self._traj[::2][-4:]
    return np.array([self._preprocess_raw_image(img).numpy() for img in last_4_images])


def train():
  env = gym.make('Breakout-v0')
  Q = DQN()
  memory = Memory(MEMORY_SIZE)
  for epoch in range(EPOCHS):
    print('Epoch {} / {}'.format(epoch, EPOCHS))
    rewards = 0
    steps = 0
    traj = Trajectory(env.reset())
    for step in tqdm(range(EPOCH_STEPS)):
      s = traj.state()
      a = Q.action(s)
      x, r, done, _ = env.step(a)
      steps += 1
      rewards += r
      traj.add(a, x)
      if step < 3: continue # traj.state needs at least 4 "x"
      memory.add(s, a, r, traj.state(), done)
      Q.optmize(memory.sample(SAMPLE_SIZE))
      if done:
        break
    print('steps = {}, rewards = {}'.format(steps, rewards))
  env.close()


if __name__ == '__main__':
  train()
