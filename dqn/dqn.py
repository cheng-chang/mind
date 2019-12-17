import random
from datetime import datetime

import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


EPOCHS = 100
EPOCH_STEPS = 10000
EPSILON = 0.05
EPSILON_MAX = 1
EPSILON_MIN = 0.1
GAMMA = 0.01
RAW_IMAGE_SHAPE = (210, 160, 3)
#PREPROCESSED_IMAGE_SHAPE = (84, 84)
PREPROCESSED_IMAGE_SHAPE = (28, 28)
#DQN_INPUT_SHAPE = (*PREPROCESSED_IMAGE_SHAPE, 4)
RAW_STATE_SHAPE = 4
DQN_INPUT_SHAPE = (RAW_STATE_SHAPE, 4)
MEMORY_SIZE = 1000
SAMPLE_SIZE = 50
#ACTIONS = 4
ACTIONS = 2
SGD_LEARNING_RATE = 1e-3
LOG_DIR="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
TENSORBOARD_CALLBACK = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, profile_batch = 3)


def image_model():
  """Build the Deep-Q-Network for raw image inputs."""
  Q = keras.Sequential()
  Q.add(layers.Conv2D(16, 8, 4, activation='relu', input_shape=DQN_INPUT_SHAPE))
  Q.add(layers.Conv2D(32, 4, 2, activation='relu'))
  Q.add(layers.Flatten())
  Q.add(layers.Dense(256, activation='relu'))
  Q.add(layers.Dense(ACTIONS))
  return Q


def vector_model():
  """Build the Deep-Q-Network for vector inputs."""
  return keras.Sequential([
    layers.Flatten(input_shape=DQN_INPUT_SHAPE),
    layers.Dense(128, activation='relu'),
    layers.Dense(ACTIONS)
  ])


class DQN:
  def __init__(self, model):
    self._Q = model
    #self._optimizer = keras.optimizers.SGD(learning_rate=SGD_LEARNING_RATE)
    self._optimizer = keras.optimizers.Adam()
    self._epsilon = EPSILON_MAX

  def reduce_epsilon(self):
    if self._epsilon > EPSILON_MIN:
      self._epsilon -= 0.001

  def _q_values(self, state):
    """Compute a forward pass of state through the Q network.

    Args:
      state is returned by Trajectory.state().

    Returns:
      a Tensor with Q values (scores) for each action, tensor shape is (4, )
    """
    #return self._Q.predict(np.array([state]), callbacks=[TENSORBOARD_CALLBACK])[0]
    #return self._Q.predict(np.array([state]))[0]
    return self._Q(np.array([state]))[0]

  def _random_action(self):
    return int(random.random() * ACTIONS)

  def _optimal_action(self, state):
    return int(tf.math.argmax(self._q_values(state)))

  def action(self, state):
    return self._random_action() if random.random() <= self._epsilon else self._optimal_action(state)

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

    Args:
      image: RAW_IMAGE_SHAPE ndarray

    Returns:
      tensor of shape PREPROCESSED_IMAGE_SHAPE
    """
    resized = tf.image.resize(image, PREPROCESSED_IMAGE_SHAPE)
    return tf.reshape(tf.image.rgb_to_grayscale(resized), PREPROCESSED_IMAGE_SHAPE)

  def _left_zero_padding(self, arr, length, shape):
    if len(arr) >= length:
      return arr
    return [np.zeros(shape)] * (length - len(arr)) + arr

  def _last_states(self):
    return self._traj[::2][-4:]

  def state(self):
    """Preprocess the last 4 raw images to a state.

    Each image is processed to a matrix,
    then stack the 4 matrices together in increasing time order.
    return Tensor with shape DQN_INPUT_SHAPE.
    """
    #last_4_images = self._left_zero_padding(self._last_states(), 4, RAW_IMAGE_SHAPE)
    #return tf.stack([self._preprocess_raw_image(img).numpy() for img in last_4_images], 2)
    return tf.stack(self._left_zero_padding(self._last_states(), 4, RAW_STATE_SHAPE))


def train():
  #env = gym.make('Breakout-v0')
  env = gym.make('CartPole-v0')
  Q = DQN(vector_model())
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
      memory.add(s, a, r, traj.state(), done)
      Q.optimize(memory.sample(SAMPLE_SIZE))
      if done:
        Q.reduce_epsilon()
        break
    print('steps = {}, rewards = {}'.format(steps, rewards))
  env.close()


if __name__ == '__main__':
  train()
