import os

import h5py
import numpy as np


DATA_DIR = "data/modelnet40_ply_hdf5_2048"
TRAIN_FILES_TXT = os.path.join(DATA_DIR, "train_files.txt")
TEST_FILES_TXT = os.path.join(DATA_DIR, "test_files.txt")


def get_lines_from_txt(path):
  """
  Args:
    path: path to a txt file

  Returns:
    list of each line in the txt file
  """
  with open(path, "r") as f:
    return [line.strip() for line in f]


def get_train_files():
  """
  Returns list of paths for train files.
  """
  return get_lines_from_txt(TRAIN_FILES_TXT)


def get_test_files():
  """
  Returns list of paths for test files.
  """
  return get_lines_from_txt(TEST_FILES_TXT)


def load_point_cloud(h5_file):
  """
  Loads the point clouds.

  Args:
    h5_file: path to the h5 file

  Returns:
    data: numpy array of shape B x N x 3
    label: numpy array of shape B x 1
  """
  f = h5py.File(h5_file)
  data = f['data'][:]
  label = f['label'][:]
  return data, label


def shuffle(data, label):
  """
  Shuffles input data and label.

  Args:
    data: point cloud, numpy array of shape B x N x 3
    label: numpy array of shape B x 1

  Returns:
    shuffled data and label, data and label still correspond to each other
  """
  idx = np.arange(data.shape[0])
  np.random.shuffle(idx)
  return data[idx, ...], label[idx]


def random_rotate_point_cloud(data):
  """
  Randomly rotate point cloud in place.

  Args:
    data: point cloud, numpy array of shape B x N x 3
  """
  for batch in range(data.shape[0]):
    points = data[batch]
    angle = np.random.uniform() * 2 * np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation_matrix = np.array([
      [cos, 0, sin],
      [0, 1, 0],
      [-sin, 0, cos]
    ])
    data[batch] = np.dot(points, rotation_matrix)


def jitter_point_cloud(data, delta=0.01, clip=0.05):
  """
  Add noise to each point in the point cloud in place.

  Args:
    data: point cloud, numpy array of shape B x N x 3
    delta: the jitter delta
    clip: the upperbound of the jitter
  """
  B, N, _ = data.shape
  jitter = np.clip(np.random.randn(B, N, 3), -1 * clip, clip)
  data += jitter


def test_load_train_file():
  files = get_train_files()
  data, label = load_point_cloud(files[0])
  print(data.shape, label.shape)


def test_load_test_file():
  files = get_test_files()
  data, label = load_point_cloud(files[0])
  print(data.shape, label.shape)


def test_shuffle():
  files = get_test_files()
  data, label = load_point_cloud(files[0])
  data, label = shuffle(data, label)
  print(data.shape, label.shape)


def test_random_rotation():
  files = get_train_files()
  data, label = load_point_cloud(files[0])
  print(data[0])
  random_rotate_point_cloud(data)
  print(data[0])


def test_jitter():
  files = get_train_files()
  data, label = load_point_cloud(files[0])
  print(data[0])
  jitter_point_cloud(data)
  print(data[0])


if __name__ == "__main__":
  test_load_train_file()
  test_load_test_file()
  test_shuffle()
  test_random_rotation()
  test_jitter()
