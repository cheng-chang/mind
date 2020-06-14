import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InputTransformNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 64, (1, 3))
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 128, 1)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 1024, 1)
    self.bn3 = nn.BatchNorm2d(1024)
    self.fc1 = nn.Linear(1024, 512)
    self.bn4 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(512, 256)
    self.bn5 = nn.BatchNorm1d(256)

  def forward(self, points):
    """
    Args:
      points: B x N x 3 Tensor: a batch B of N 3-dimensional points

    Returns:
      B x 3 x 3 Tensor: a batch B of 3 x 3 transformation matrix
    """
    B, N, _ = points.shape

    x = points.unsqueeze(1)
    # B x 1 x N x 3
    x = F.relu(self.bn1(self.conv1(x)))
    # B x 64 x N x 1
    x = F.relu(self.bn2(self.conv2(x)))
    # B x 128 x N x 1
    x = F.relu(self.bn3(self.conv3(x)))
    # B x 1024 x N x 1
    x = F.max_pool2d(x, (N, 1))
    # B x 1024 x 1 x 1
    x = x.reshape((B, -1))
    # B x 1024
    x = F.relu(self.bn4(self.fc1(x)))
    # B x 512
    x = F.relu(self.bn5(self.fc2(x)))
    # B x 256

    # w: 256 x 9
    # b: 9
    # xw + b: B x 9 -> B x 3 x 3
    w = torch.zeros(256, 9, requires_grad=True).to(DEVICE)
    b = torch.eye(3, requires_grad=True).to(DEVICE).flatten()
    t = torch.mm(x, w) + b
    return t.reshape((B, 3, 3))


class FeatureTransformNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 64, (1, 64))
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 128, 1)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 1024, 1)
    self.bn3 = nn.BatchNorm2d(1024)
    self.fc1 = nn.Linear(1024, 512)
    self.bn4 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(512, 256)
    self.bn5 = nn.BatchNorm1d(256)

  def forward(self, features):
    """
    Args:
      features: B x N x 64 Tensor representing a batch B of N 64-dimensional features

    Returns:
      B x 64 x 64 Tensor: a batch B of 64 x 64 transformation matrix
    """
    B, N, _ = features.shape

    x = features.unsqueeze(1)
    # B x 1 x N x 64
    x = F.relu(self.bn1(self.conv1(x)))
    # B x 64 x N x 1
    x = F.relu(self.bn2(self.conv2(x)))
    # B x 128 x N x 1
    x = F.relu(self.bn3(self.conv3(x)))
    # B x 1024 x N x 1
    x = F.max_pool2d(x, (N, 1))
    # B x 1024 x 1 x 1
    x = x.reshape((B, -1))
    # B x 1024
    x = F.relu(self.bn4(self.fc1(x)))
    # B x 512
    x = F.relu(self.bn5(self.fc2(x)))
    # B x 256

    # w: 256 x (64x64)
    # b: (64x64)
    # xw + b: B x (64x64) -> B x 64 x64
    w = torch.zeros(256, 64 * 64, requires_grad=True).to(DEVICE)
    b = torch.eye(64, requires_grad=True).to(DEVICE).flatten()
    t = torch.mm(x, w) + b
    return t.reshape((B, 64, 64))


class PointNetClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_transform_net = InputTransformNet()
    self.feature_transform_net = FeatureTransformNet()
    self.conv1 = nn.Conv2d(1, 64, (1, 3))
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 64, 1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(1, 64, (1, 64))
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, 1)
    self.bn4 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(128, 1024, 1)
    self.bn5 = nn.BatchNorm2d(1024)
    self.fc1 = nn.Linear(1024, 512)
    self.bn6 = nn.BatchNorm1d(512)
    self.dp1 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(512, 256)
    self.bn7 = nn.BatchNorm1d(256)
    self.dp2 = nn.Dropout(0.3)
    self.fc3 = nn.Linear(256, 40)
    self.bn8 = nn.BatchNorm1d(40)

  def forward(self, points):
    """
    Each object is represented as point clouds,
    classify the objects into 40 classes.

    Args:
      points: B x N x 3 Tensor, B is batch size, N is number of points

    Returns:
      1. B x 40 Tensor, v = [a1, a2, ..., a40], ai / sum(ai) is the predicted
      probability that the object is of class i, argmax(v) is the predicted class
      2. B x 64 x 64 feature transform matrix,
      will be regularized to be orthogonal during optimization
    """
    B, N, _ = points.shape

    input_transform_matrix = self.input_transform_net(points)
    points = torch.bmm(points, input_transform_matrix)
    x = points.unsqueeze(1)
    # B x 1 x N x 3
    x = F.relu(self.bn1(self.conv1(x)))
    # B x 64 x N x 1
    x = F.relu(self.bn2(self.conv2(x)))
    # B x 64 x N x 1
    x = x.squeeze(-1)
    # B x 64 x N
    x = x.transpose(1, 2)
    # B x N x 64
    feature_transform_matrix = self.feature_transform_net(x)
    x = torch.bmm(x, feature_transform_matrix)
    # B x N x 64
    x = x.unsqueeze(1)
    # B x 1 x N x 64
    x = F.relu(self.bn3(self.conv3(x)))
    # B x 64 x N x 1
    x = F.relu(self.bn4(self.conv4(x)))
    # B x 128 x N x 1
    x = F.relu(self.bn5(self.conv5(x)))
    # B x 1024 x N x 1
    x = F.max_pool2d(x, (N, 1))
    # B x 1024 x 1 x 1
    x = x.reshape((B, -1))
    # B x 1024
    x = F.relu(self.bn6(self.fc1(x)))
    x = self.dp1(x)
    # B x 512
    x = F.relu(self.bn7(self.fc2(x)))
    x = self.dp2(x)
    # B x 256
    x = F.relu(self.bn8(self.fc3(x)))
    # B x 40
    return x, feature_transform_matrix


def test_classifier():
  B = 2
  N = 2
  model = PointNetClassifier()
  points = torch.randn(B, N, 3)
  classes, feature_transform_matrix = model(points)
  print(classes)
  print(classes.shape)
  print(feature_transform_matrix)
  print(feature_transform_matrix.shape)


if __name__ == '__main__':
  test_classifier()
