import model as model_module
import data as data_module

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, 'log')
LOGGER = SummaryWriter(LOG_DIR)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCH = 250
BATCH = 32
BASE_LR = 0.001
LR_DECAY_EPOCH = 13
LR_DECAY_RATE = 0.8


def regularize_orthogonal_matrix(matrix, weight=0.001):
  """
  Regularizes the matrix to be orthogonal.

  Args:
    matrix: the matrix to be regularized, torch.Tensor of shape B x K x K
    weight: the regularization weight

  Returns:
    the regularization loss, scalar
  """
  B, K, _ = matrix.shape
  mm = torch.bmm(matrix, torch.transpose(matrix, 1, 2))
  I = torch.eye(K).to(DEVICE)
  I = I.reshape((1, K, K))
  I = I.repeat(B, 1, 1)
  diff = mm - I
  l2 = nn.MSELoss()
  loss = l2(diff, torch.zeros(diff.shape).to(DEVICE))
  return loss * weight


def accuracy(pred, label):
  """
  Computes accuracy of prediction.

  Args:
    pred: B x C Tensor, B is batch size, C is number of classes
    label: B x 1 Tensor

  Returns:
    percentage of correct predictions
  """
  pred_class = np.argmax(pred.cpu().detach().numpy(), 1)
  correct = np.sum(pred_class == label.cpu().numpy())
  accuracy = correct / label.shape[0]
  return accuracy


def train_one_batch(batch_index, optimizer, model, data, label):
  pred, feature_transform_matrix = model(data)
  # loss
  label = torch.squeeze(label)
  cross_entropy = nn.CrossEntropyLoss()
  loss = cross_entropy(pred, label) + \
         regularize_orthogonal_matrix(feature_transform_matrix)
  if batch_index % 100 == 99:
    LOGGER.add_scalar('train loss', loss.item(), batch_index)
    LOGGER.add_scalar('train accuracy', accuracy(pred, label), batch_index)
  # optimize
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


def eval_one_batch(batch_index, model, data, label):
  with torch.no_grad():
    pred, _ = model(data)
    label = torch.squeeze(label)
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(pred, label)
    if batch_index % 100 == 99:
      LOGGER.add_scalar('eval loss', loss.item(), batch_index)
      LOGGER.add_scalar('eval accuracy', accuracy(pred, label), batch_index)


def yield_batch(file):
  """
  Yields batches of (data, label) as pair of Tensors.
  """
  data, label = data_module.load_point_cloud(file)
  data, label = data_module.shuffle(data, label)
  data_module.random_rotate_point_cloud(data)
  data_module.jitter_point_cloud(data)
  num_batches = data.shape[0] // BATCH
  for batch in range(num_batches):
    start_idx = batch * BATCH
    end_idx = (batch + 1) * BATCH
    data_batch = torch.tensor(data[start_idx:end_idx, :, :]).to(DEVICE)
    label_batch = torch.tensor(label[start_idx:end_idx, :], dtype=torch.long).to(DEVICE)
    yield data_batch, label_batch


def train_one_epoch(optimizer, model):
  train_files = data_module.get_train_files()
  random.shuffle(train_files)
  batch_index = 0
  for file in train_files:
    for data, label in yield_batch(file):
      train_one_batch(batch_index, optimizer, model, data, label)
      batch_index += 1


def eval_one_epoch(model):
  test_files = data_module.get_test_files()
  batch_index = 0
  for file in test_files:
    for data, label in yield_batch(file):
      eval_one_batch(batch_index, model, data, label)
      batch_index += 1


def train():
  model = model_module.PointNetClassifier()
  model.to(DEVICE)
  optimizer = optim.Adam(model.parameters(), lr=BASE_LR)
  lr_sched = optim.lr_scheduler.StepLR(optimizer, LR_DECAY_EPOCH, LR_DECAY_RATE)
  for epoch in range(EPOCH):
    train_one_epoch(optimizer, model)
    eval_one_epoch(model)
    lr_sched.step()
    if epoch % 10 == 0:
      torch.save(model.state_dict(), os.path.join(LOG_DIR, "model.ckpt"))


if __name__ == '__main__':
  train()
