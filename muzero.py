import os
import time
from datetime import datetime
import threading
from multiprocessing import Process, Queue, Value
import ctypes
import math
import typing

from tqdm import tqdm
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


QUIT = "quit"
FLOAT_MAX = float('inf')
EPOCHS = int(1e6)
EPOCH_STEPS = 200
PLAN_STEPS = 80
ACTORS = 3
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 32
ACTIONS = 2
ROOT_PRIOR_NOISE_DIRICHLET_ALPHA = 0.03
ROOT_PRIOR_NOISE_FRACTION = 0.25
UPPER_CONFIDENCE_BOUND_C1 = 1.25
UPPER_CONFIDENCE_BOUND_C2 = 19652
DISCOUNT = 0.997

OBSERVATION_SIZE = 4
HIDDEN_NEURON_SIZE = 64
HIDDEN_STATE_SIZE = 4

# training parameters
UNROLLED_STEPS = 5
DISCOUNTED_VALUE_STEPS = 10
LEARNING_RATE_BASE = 0.05
LEARNING_RATE_DECAY_STEPS = 350e3
LEARNING_RATE_DECAY_RATE = 0.1
MODEL_CHECKPOINT_INTERVAL = 1000
L2_REGULARIZER_WEIGHT = 1e-4


def timestamp():
  return datetime.now().strftime("%Y%m%d-%H%M%S")


class RepresentationNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(OBSERVATION_SIZE, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, HIDDEN_STATE_SIZE)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    return x


class DynamicsNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE + 1, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, HIDDEN_STATE_SIZE)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    return x


class RewardNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE + 1, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class PolicyNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, ACTIONS)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class ValueNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class MuZeroNetOutput(typing.NamedTuple):
  state: torch.Tensor
  reward: torch.Tensor
  policy_logits: torch.Tensor
  value: torch.Tensor


class DummyMuZeroNet:
  def represent(self, observation):
    return MuZeroNetOutput(torch.randn_like(observation), \
      torch.tensor([0.]), \
      torch.tensor([[1., 1.]]), \
      torch.tensor([0.]))

  def transit(self, state, action):
    return MuZeroNetOutput(torch.randn_like(state), \
      torch.tensor([0.]), \
      torch.tensor([[1., 1.]]), \
      torch.tensor([0.]))

  def training_steps(self):
    return 0


class MuZeroNet:
  def __init__(self):
    self._representation_net = RepresentationNet()
    self._dynamics_net = DynamicsNet()
    self._reward_net = RewardNet()
    self._policy_net = PolicyNet()
    self._value_net = ValueNet()
    params = list(self._representation_net.parameters()) + \
      list(self._dynamics_net.parameters()) + \
      list(self._reward_net.parameters()) + \
      list(self._policy_net.parameters()) + \
      list(self._value_net.parameters())
    self._optimizer = optim.Adam(params, lr=LEARNING_RATE_BASE, weight_decay=L2_REGULARIZER_WEIGHT)
    self._lr_scheduler = optim.lr_scheduler.StepLR(self._optimizer, \
      step_size = LEARNING_RATE_DECAY_STEPS, \
      gamma = LEARNING_RATE_DECAY_RATE)

  def represent(self, observation) -> MuZeroNetOutput:
    """Transforms the observation into hidden state space.

    Args:
      observation (torch tensor): minibatch of environment observations, size should be [batch_size, observation_size].
    """
    state = self._representation_net(observation)
    policy_logits = self._policy_net(state)
    value = self._value_net(state)
    zero_rewards = torch.zeros((state.size()[0], 1))
    return MuZeroNetOutput(state, zero_rewards, policy_logits, value)

  def transit(self, state, action) -> MuZeroNetOutput:
    """Transits from hidden state with the specified action.

    Args:
      state (torch tensor): minibatch of hidden states, size should be [batch_size, hidden_state_size].
      action (torch tensor): minibatch of actions, size should be [batch_size].
    """
    state_action = torch.cat((state, action.reshape((-1, 1)).float()), -1)
    next_state = self._dynamics_net(state_action)
    reward = self._reward_net(state_action)
    policy_logits = self._policy_net(next_state)
    value = self._value_net(next_state)
    return MuZeroNetOutput(next_state, reward, policy_logits, value)

  def training_steps(self):
    return self._lr_scheduler.last_epoch + 1

  def train(self, batch):
    """Trains all networks end-to-end.

    batch is sampled from Memory.sample.
    """
    observation_batch, action_batch, target_reward, target_policy, target_value = (t.float() for t in map(torch.tensor, batch))
    batch_size = len(observation_batch)
    unrolled_steps = len(target_reward[0])

    state, _, policy_logits, value = self.represent(observation_batch)
    reward_loss = torch.zeros(batch_size)
    policy_loss = self._policy_loss(policy_logits, target_policy, 0)
    value_loss = self._value_loss(value, target_value, 0)

    gradient_scale = 1 / unrolled_steps
    for step in range(unrolled_steps):
      state, reward, policy_logits, value = self.transit(state, action_batch[:, step])
      reward_loss += gradient_scale * self._reward_loss(reward, target_reward, step)
      policy_loss += gradient_scale * self._policy_loss(policy_logits, target_policy, step + 1)
      value_loss += gradient_scale * self._value_loss(value, target_value, step + 1)

    loss = reward_loss + policy_loss + value_loss
    loss = loss.mean()
    print("{}: step={}, loss={}".format(timestamp(), self.training_steps(), loss))

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()
    self._lr_scheduler.step()

  def _policy_loss(self, policy_logits, target_policy, step_idx):
    return self._cross_entropy_logits_loss(policy_logits, target_policy[:, step_idx])

  def _value_loss(self, value, target_value, step_idx):
    return self._scalar_loss(value.squeeze(-1), target_value[:, step_idx])

  def _reward_loss(self, reward, target_reward, step_idx):
    return self._scalar_loss(reward.squeeze(-1), target_reward[:, step_idx])

  def _scalar_loss(self, pred, target):
    return nn.MSELoss(reduction='none')(pred, target)

  def _cross_entropy_logits_loss(self, logits, probs):
    return (-torch.log_softmax(logits, -1) * probs).sum(-1)


class NetStorage:
  """Stores neural networks to be shared among Actors."""
  def put(self, net):
    """Stores the net as the latest."""
    raise NotImplementedError

  def get(self, net=None):
    """Retrieves the latest stored network.

    1. If there is no stored network, and net is not None, then return net;
    2. If there is no stored network, and net is None, then return a default one.
    3. If net is provided and its training step is larger than or
    equal to the latest stored network, then return the provided net.
    4. Otherwise, return the latest stored network.
    """
    raise NotImplementedError


class PersistedNetStorage(NetStorage):
  """Stores the network on disk.

  Each stored network is an on-disk file named "{training step}.model"
  under the specified model directory.
  """
  def __init__(self, model_dir):
    self._model_dir = model_dir
    self._latest_step = Value('i', -1)

  def _model_path(self, step):
    return os.path.join(self._model_dir, '{}.model'.format(step))

  def put(self, net):
    step = net.training_steps()
    torch.save(net, self._model_path(step))
    self._latest_step.value = step

  def get(self, net=None):
    latest_step = self._latest_step.value
    if latest_step == -1:
      return net if net else DummyMuZeroNet()
    if net and net.training_steps() >= latest_step:
      return net
    return torch.load(self._model_path(latest_step))


class Planner:
  def plan(self, observation):
    """Plan the possible future trajectories from observations and return the action to take."""
    raise NotImplementedError


class MinMax:
  """Keep minimum and maximum values seen so far."""
  def __init__(self):
    self._min = FLOAT_MAX
    self._max = -FLOAT_MAX

  def update(self, value: float):
    self._min = min(self._min, value)
    self._max = max(self._max, value)

  def normalize(self, value: float):
    if self._max <= self._min:
      return value
    return (value - self._min) / (self._max - self._min)


class Node:
  def __init__(self, netoutput: MuZeroNetOutput, parent = None, minmax: MinMax = None):
    self._minmax = minmax if minmax is not None else MinMax()
    self._parent = parent
    self._reward = netoutput.reward[0].item()
    self._state = netoutput.state[0]
    self._value_sum = netoutput.value[0].item()
    self._visit_count = 1
    self._action_prior = self._compute_action_prior(netoutput.policy_logits[0], parent is None)
    self._children = {}

  @staticmethod
  def _compute_action_prior(policy_logits, should_add_noise):
    policy = torch.softmax(policy_logits, -1)
    prior = {a: policy[a].item() for a in range(ACTIONS)}
    if should_add_noise:
      prior = Node._add_dirichlet_prior_noise(prior)
    return prior

  @staticmethod
  def _add_dirichlet_prior_noise(prior):
    noise = np.random.dirichlet([ROOT_PRIOR_NOISE_DIRICHLET_ALPHA] * ACTIONS)
    frac = ROOT_PRIOR_NOISE_FRACTION
    return {a: (frac * noise[a] + (1 - frac) * prior[a]) for a in range(ACTIONS)}

  def _q_value(self, action):
    if action not in self._children:
      return 0
    child = self._children[action]
    return child._reward + DISCOUNT * child.value()

  def action_count(self, action):
    return self._children[action]._visit_count if action in self._children else 0

  def state(self):
    return self._state

  def policy(self):
    counts = np.array([self.action_count(a) for a in range(ACTIONS)])
    return counts / sum(counts)

  def value(self):
    return self._value_sum / self._visit_count

  def choose_action(self):
    _, a = max((self._upper_confidence_bound(a), a) for a in range(ACTIONS))
    return a

  def _upper_confidence_bound(self, action):
    action_counts = self._visit_count
    action_count = self.action_count(action)
    action_prior = self._action_prior[action]
    c1 = UPPER_CONFIDENCE_BOUND_C1
    c2 = UPPER_CONFIDENCE_BOUND_C2
    prior_weight = math.sqrt(action_counts) / (1 + action_count)
    prior_weight *= (c1 + math.log((action_counts + c2 + 1) / c2))
    prior_score = action_prior * prior_weight

    value_score = self._minmax.normalize(self._q_value(action))

    return value_score + prior_score

  def has_child(self, action):
    return action in self._children

  def get_child(self, action):
    return self._children[action]

  def add_child(self, action, netoutput: MuZeroNetOutput):
    child = Node(netoutput, self, self._minmax)
    self._children[action] = child
    child._backtrack(netoutput.value[0].item())

  def _backtrack(self, value):
    if self._parent is None:
      return
    value = self._reward + DISCOUNT * value
    self._value_sum += value
    self._visit_count += 1
    self._minmax.update(value)
    self._parent._backtrack(value)


class MonteCarloTreeSearch(Planner):
  def __init__(self, muzero: MuZeroNet):
    self._muzero = muzero
    self._root = None

  def plan(self, observation):
    self._root = Node(self._muzero.represent(torch.tensor([observation], dtype=torch.float)))
    for _ in range(PLAN_STEPS):
      self._plan(self._root)
    return self._choose_action(self._root)

  def root(self):
    return self._root

  def _plan(self, root):
    node = root
    while True:
      a = node.choose_action()
      if node.has_child(a):
        node = node.get_child(a)
      else:
        break
    action = node.choose_action()
    output = self._muzero.transit(node.state().unsqueeze(0), torch.tensor(action))
    node.add_child(action, output)

  def _choose_action(self, root):
    actions = list(range(ACTIONS))
    counts = np.array([root.action_count(a) for a in actions])
    t = self._action_count_temperature()
    idx = self._sample_index(counts, t)
    return actions[idx]

  def _action_count_temperature(self):
    steps = self._muzero.training_steps()
    if steps <= 5e5:
      return 1
    if steps <= 7.5e5:
      return 0.5
    return 0.25

  @staticmethod
  def _sample_index(counts, temperature):
    counts = counts ** (1 / temperature)
    probs = counts / sum(counts)
    return np.random.choice(len(counts), p=probs)


class Trajectory:
  """A trajectory is (x0, a0, r0, p0, v0, x1, a1, r1, p1, v1, ..., xn).

  x (numpy array) is the raw observation from the environment,
  a (int) is action,
  r (float) is the immediate reward of the previous action,
  p (numpy array) is the probability of actions to be chosen from x,
  v (float) is the value estimate of x.

  NOTE: x will be transformed to a state representation in the hidden space,
  the transformed state uses symbol s.
  """
  def __init__(self, x0):
    self._observations = [x0]
    self._actions = []
    self._rewards = []
    self._policies = []
    self._values = []

  def last_observation(self):
    return self._observations[-1]

  def append(self, current_action, reward, policy, value, next_observation):
    self._actions.append(current_action)
    self._rewards.append(reward)
    self._policies.append(policy)
    self._values.append(value)
    self._observations.append(next_observation)

  def _max_sample_size(self):
    return len(self._actions)

  def _sample_start_index(self):
    return np.random.choice(range(self._max_sample_size()))

  def sample(self, unrolled_steps):
    """Returns a sample in Memory.sample.
    observation (numpy array)
    action (int)
    reward (float)
    policy (numpy array)
    value (float)
    """
    start_index = self._sample_start_index()
    end_index = start_index + unrolled_steps
    # actions, rewards: [start_index, end_index)
    # policies, values: [start_index, end_index]
    observation = self._observations[start_index]
    actions = np.array(self._actions[start_index : end_index + 1], dtype=np.float)
    rewards = np.array(self._rewards[start_index : end_index + 1], dtype=np.float)
    policies = np.array(self._policies[start_index : end_index + 1], dtype=np.float)
    for _ in range(end_index + 1 - self._max_sample_size()):
      actions = np.append(actions, np.random.choice(range(ACTIONS)))
      rewards = np.append(rewards, 0)
      policies = np.append(policies, [np.zeros(policies[0].shape)], axis=0)
    actions = actions[:-1]
    rewards = rewards[:-1]
    values = self._discounted_values(start_index, end_index)
    return (observation, actions, rewards, policies, values)

  def _discounted_values(self, start_index, end_index):
    i = end_index + DISCOUNTED_VALUE_STEPS
    value = self._value(i)
    i -= 1
    while i >= end_index:
      value = self._reward(i) + DISCOUNT * value
      i -= 1
    values = [value]

    value_frac = DISCOUNT ** DISCOUNTED_VALUE_STEPS
    last_reward_frac = value_frac / DISCOUNT
    while i >= start_index:
      last_i = i + DISCOUNTED_VALUE_STEPS + 1
      value -= value_frac * self._value(last_i)
      value -= last_reward_frac * self._reward(last_i - 1)
      value *= DISCOUNT
      value += value_frac * self._value(last_i - 1)
      value += self._reward(i)
      i -= 1
      values.append(value)
    values.reverse()
    return np.array(values, dtype=np.float)

  def _value(self, idx):
    return self._values[idx] if idx < len(self._values) else 0

  def _reward(self, idx):
    return self._rewards[idx] if idx < len(self._rewards) else 0


class Memory:
  def __init__(self):
    self._lock = threading.Lock()
    self._trajectories = []

  def store(self, trajectory: Trajectory):
    """Store trajectory."""
    with self._lock:
      if len(self._trajectories) >= MEMORY_SIZE:
        self._trajectories.pop(0)
      self._trajectories.append(trajectory)

  def size(self):
    with self._lock:
      return len(self._trajectories)

  def sample(self, batch_size):
    """Returns (observation_batch, actions_batch, rewards_batch, policies_batch, values_batch).

    Each batch is a numpy array.
    The size of the returned batch can be < batch_size, could be 0.

    self.size() must > batch_size.
    """
    with self._lock:
      observation_batch = []
      actions_batch = []
      rewards_batch = []
      policies_batch = []
      values_batch = []
      for _ in range(batch_size):
        t = np.random.choice(self._trajectories)
        observation, actions, rewards, policies, values = t.sample(UNROLLED_STEPS)
        observation_batch.append(observation)
        actions_batch.append(actions)
        rewards_batch.append(rewards)
        policies_batch.append(policies)
        values_batch.append(values)
      return tuple(map(np.array, (observation_batch, actions_batch, rewards_batch, policies_batch, values_batch)))


def play(net_storage, queue, kill_signal):
  """Reset the environment and collect a new trajectory into queue."""
  net = None
  while (not kill_signal.value) and (net is None or net.training_steps() < EPOCHS):
    net = net_storage.get(net)
    play_one_epoch(net, queue)


def play_one_epoch(net, queue):
  env = gym.make('CartPole-v0')
  trajectory = Trajectory(env.reset())
  steps = 0
  #for _ in tqdm(range(EPOCH_STEPS)):
  for _ in range(EPOCH_STEPS):
    steps += 1
    observation = trajectory.last_observation()
    planner = MonteCarloTreeSearch(net)
    action = planner.plan(observation)
    next_observation, reward, done, _ = env.step(action)
    root = planner.root()
    policy = root.policy()
    value = root.value()
    trajectory.append(action, reward, policy, value, next_observation)
    if done:
      break
  print("steps={}".format(steps))
  queue.put(trajectory)


class Actors:
  def start(self):
    raise NotImplementedError

  def join(self):
    raise NotImplementedError


class ThreadActors(Actors):
  def __init__(self, num_actors, fn, args):
    self._threads = []
    for _ in range(num_actors):
      self._threads.append(threading.Thread(target=fn, args=args))

  def start(self):
    for t in self._threads:
      t.start()

  def join(self):
    for t in self._threads:
      t.join()


class MultiProcessActors(Actors):
  def __init__(self, num_actors, fn, args):
    self._processes = []
    for _ in range(num_actors):
      self._processes.append(Process(target=fn, args=args, daemon=True))

  def start(self):
    for p in self._processes:
      p.start()

  def join(self):
    for p in self._processes:
      p.join()


def train(muzero_net, memory, net_storage):
  while memory.size() < BATCH_SIZE: time.sleep(10)
  print('memory size = {}'.format(memory.size()))
  muzero_net.train(memory.sample(BATCH_SIZE))
  training_steps = muzero_net.training_steps()
  if training_steps > 0 and training_steps % MODEL_CHECKPOINT_INTERVAL == 0:
    net_storage.put(muzero_net)


def transfer(queue, memory):
  """Transfers trajectory from queue to memory."""
  while True:
    trajectory = queue.get()
    if trajectory == QUIT:
      break
    memory.store(trajectory)


def main():
  try:
    muzero_net = MuZeroNet()

    experiment_dir = os.path.join(os.getcwd(), 'experiments', timestamp())
    model_dir = os.path.join(experiment_dir, 'models')
    os.makedirs(model_dir)

    net_storage = PersistedNetStorage(model_dir)
    queue = Queue(ACTORS * BATCH_SIZE)
    kill_signal = Value(ctypes.c_bool, False)
    actors = MultiProcessActors(ACTORS, play, (net_storage, queue, kill_signal))
    actors.start()

    memory = Memory()
    transferer = threading.Thread(target=transfer, args=(queue, memory))
    transferer.start()

    for epoch in range(EPOCHS):
      print('Epoch {}'.format(epoch))
      train(muzero_net, memory, net_storage)
      time.sleep(1)
    net_storage.put(muzero_net)
  finally:
    queue.put(QUIT)
    transferer.join()

    kill_signal.value = True
    actors.join()

    if len(os.listdir(model_dir)) == 0:
      os.removedirs(model_dir)


if __name__ == '__main__':
  main()
