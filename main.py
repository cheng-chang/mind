import random

import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


EPOCHS = 10000
EPSILON = 0.01
PREPROCESSED_IMAGE_SHAPE = (110, 84)
DQN_INPUT_SHAPE = (*PREPROCESSED_IMAGE_SHAPE, 4)
MEMORY_SIZE = 100
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


def preprocess_history(history):
  pass


def build_dqn():
  """Build the Deep-Q-Network."""
  Q = keras.Sequential()
  Q.add(layers.Conv2D(16, (8, 8), 4, activation='relu', input_shape=DQN_INPUT_SHAPE))
  Q.add(layers.Conv2D(32, (4, 4), 2, activation='relu'))
  Q.add(layers.Flatten())
  Q.add(layers.Dense(256, activation='relu'))
  Q.add(layers.Dense(ACTIONS, activation='softmax'))
  return Q


def collect_history(env, Q):
  pass


def show_progress():
  pass


def train():
  env = gym.make('Breakout-v0')
  env.reset()
  for _ in range(10000000):
    env.render()
    o, r, done, _ = env.step(a)
    if done:
      env.reset()
  env.close()


if __name__ == '__main__':
  train()