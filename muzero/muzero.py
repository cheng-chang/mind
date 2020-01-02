import threading
import math
import typing

from tqdm import tqdm
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FLOAT_MAX = float('inf')
EPOCHS = int(1e6)
EPOCH_STEPS = 200
PLAN_STEPS = 80
ACTORS = 10
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 64
ACTIONS = 2
ROOT_PRIOR_NOISE_DIRICHLET_ALPHA = 0.03
ROOT_PRIOR_NOISE_FRACTION = 0.25
UPPER_CONFIDENCE_BOUND_C1 = 1.25
UPPER_CONFIDENCE_BOUND_C2 = 19652
DISCOUNT = 0.997

RAW_STATE_SIZE = 4
HIDDEN_NEURON_SIZE = 64
HIDDEN_STATE_SIZE = 4

# training parameters
UNROLLED_STEPS = 5


class RepresentationNet(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(RAW_STATE_SIZE, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, HIDDEN_STATE_SIZE)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.tanh(self.fc2(x))
    return x


class DynamicsNet(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE + 1, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, HIDDEN_STATE_SIZE)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.tanh(self.fc2(x))
    return x


class RewardNet(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE + 1, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class PolicyNet(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, ACTIONS)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class ValueNet(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(HIDDEN_STATE_SIZE, HIDDEN_NEURON_SIZE)
    self.fc2 = nn.Linear(HIDDEN_NEURON_SIZE, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


class MuZeroNet:
  def __init__(self):
    self._representation_net = RepresentationNet()
    self._dynamics_net = DynamicsNet()
    self._reward_net = RewardNet()
    self._policy_net = PolicyNet()
    self._value_net = ValueNet()
    self._training_steps = 0

  def represent(self, raw_state) -> MuZeroNetOutput:
    state = self._representation_net(raw_state)
    policy_logits = self._policy_net(state)
    value = self._value_net(state)
    return MuZeroNetOutput(state, 0, policy_logits, value)

  def transit(self, state, action) -> MuZeroNetOutput:
    state_action = np.append(state, action)
    next_state = self._dynamics_net(state_action)
    reward = self._reward_net(state_action)
    policy_logits = self._policy_net(next_state)
    value = self._value_net(next_state)
    return MuZeroNetOutput(next_state, reward, policy_logits, value)

  def train(self, trajectories):
    self._training_steps += 1
    pass

  def training_steps(self):
    return self._training_steps


class MuZeroNetOutput(typing.NamedTuple):
  state: typing.List[float]
  reward: float
  policy_logits: typing.Dict[int, float]
  value: float


class Planner:
  def plan(self, state):
    """Plan the possible future trajectories from raw state and return the action to take."""
    raise NotImplementedError


class ActionStats:
  def __init__(self, prior):
    self._visit_count = 0
    self._q_value_sum = 0
    self._prior = prior

  def update(self, q_value):
    self._visit_count += 1
    self._q_value_sum += q_value

  def visit_count(self):
    return self._visit_count

  def prior(self):
    return self._prior

  def q_value(self):
    return 0 if self._visit_count == 0 else self._q_value_sum / self._visit_count


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
  def __init__(self, netoutput: MuZeroNetOutput, parent: Node = None, action = None, minmax: MinMax = None):
    self._minmax = minmax if minmax is not None else MinMax()
    self._parent = parent
    self._action = action
    self._reward = netoutput.reward
    self._state = netoutput.state
    prior = self._compute_action_prior(netoutput.policy_logits, parent is None)
    self._action_stats = {a: ActionStats(prior[a]) for a in ACTIONS}
    self._children = {}

  @staticmethod
  def _compute_action_prior(policy_logits, should_add_noise):
    policy = {a: math.exp(policy_logits[a]) for a in ACTIONS}
    policy_sum = sum(policy.values())
    prior = {a: policy[a] / policy_sum for a in ACTIONS}
    if should_add_noise:
      prior = Node._add_dirichlet_prior_noise(prior)
    return prior

  @staticmethod
  def _add_dirichlet_prior_noise(prior):
    noise = np.random.dirichlet([ROOT_PRIOR_NOISE_DIRICHLET_ALPHA] * ACTIONS)
    frac = ROOT_PRIOR_NOISE_FRACTION
    return {a: (frac * noise[a] + (1 - frac) * prior[a]) for a in ACTIONS}

  def state(self):
    return self._state

  def choose_action(self):
    _, a = max((self._upper_confidence_bound(a), a) for a in ACTIONS)
    return a

  def _upper_confidence_bound(self, action):
    stats = self._action_stats
    stat = stats[action]
    action_counts = sum(map(lambda a: stats[a].visit_count(), stats))
    action_count = stat.visit_count()
    c1 = UPPER_CONFIDENCE_BOUND_C1
    c2 = UPPER_CONFIDENCE_BOUND_C2

    prior_weight = math.sqrt(action_counts) / (1 + action_count)
    prior_weight *= (c1 + math.log((action_counts + c2 + 1) / c2))
    prior_score = stat.prior() * prior_weight

    value_score = self._minmax.normalize(stat.q_value())

    return value_score + prior_score

  def has_child(self, action):
    return action in self._children

  def get_child(self, action):
    return self._children[action]

  def add_child(self, action, netoutput: MuZeroNetOutput):
    child = Node(netoutput, self, action, self._minmax)
    self._children[action] = child
    child._backtrack(netoutput.value)

  def _backtrack(self, value):
    if self._parent is None:
      return
    value = self._reward + DISCOUNT * value
    stat = self._parent._action_stats[self._action]
    stat.update(value)
    self._minmax.update(stat.q_value())
    self._parent._backtrack(value)

  def action_counts(self):
    return {action: stat.visit_count() for (action, stat) in self._action_stats.items()}

  def action_probs(self):
    counts = self.action_counts()
    counts_sum = sum(counts.values())
    return np.array([counts[a] / counts_sum for a in range(ACTIONS)])

  def _value(self, action):
    return self._action_stats[action].q_value()

  def action_values(self):
    return np.array([self._value(a) for a in range(ACTIONS)])


class MonteCarloTreeSearch(Planner):
  def __init__(self, muzero: MuZeroNet):
    self._muzero = muzero
    self._root = None

  def plan(self, state):
    self._root = Node(self._muzero.represent(state))
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
    output = self._muzero.transit(node.state(), action)
    node.add_child(action, output)

  def _choose_action(self, root):
    action_counts = root.action_counts()
    actions = action_counts.keys()
    counts = np.array([action_counts[a] for a in actions])
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

  x is the raw observation from the environment,
  a is action,
  r is the immediate reward of the previous action,
  p is the probability of actions to be chosen from x,
  v is the value estimate of x.

  NOTE: x will be transformed to a state representation in the hidden space,
  the transformed state uses symbol s.
  """
  def __init__(self, x0):
    self._states = [x0]
    self._actions = []
    self._rewards = []
    self._policies = []
    self._values = []

  def last_state(self):
    return self._states[-1]

  def append(self, current_action, reward, policy, value, next_state):
    self._actions.append(current_action)
    self._rewards.append(reward)
    self._policies.append(policy)
    self._values.append(value)
    self._states.append(next_state)

  def num_states(self):
    return len(self._states)

  def sample(self, init_state_index, unrolled_steps):
    """Returns a sample in Memory.sample."""
    end_index = init_state_index + unrolled_steps
    state = self._states[init_state_index]
    actions = tuple(self._actions[init_state_index : end_index])
    rewards = tuple(self._rewards[init_state_index : end_index])
    policies = tuple(self._policies[init_state_index : end_index])
    values = tuple(self._values[init_state_index : end_index])
    return (state, actions, (rewards, policies, values))


class Memory:
  def __init__(self):
    """Might be called concurrently by multiple actors, need to be thread-safe."""
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
    """Returns a batch of (raw_state, actions, targets) where raw_state
    is uniformly sampled from a uniformly sampled trajectory.

    targets = (rewards, policies, values)
    Each value is a discounted sum of future rewards.
    actions, rewards, policies, values are all tuples.
    len(actions) == len(targets)
    len(actions) <= UNROLLED_STEPS

    self.size() must > batch_size.
    """
    with self._lock:
      batch = []
      for _ in range(batch_size):
        traj = np.random.choice(self._trajectories)
        n = traj.num_states()
        idx = np.random.choice(range(traj.num_states() - 1))
        batch.append(traj.sample(idx, UNROLLED_STEPS))
      return batch


class Actors:
  def run(self):
    raise NotImplementedError


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


def play(env, muzero_net, memory):
  """Reset the environment and collect a new trajectory into memory."""
  planner = MonteCarloTreeSearch(muzero_net)
  trajectory = Trajectory(env.reset())
  for _ in tqdm(range(EPOCH_STEPS)):
    x = trajectory.last_state()
    a = planner.plan(x)
    nx, r, done, _ = env.step(a)
    root = planner.root()
    policy = root.action_probs()
    values = root.action_values()
    value = np.dot(policy, values)
    trajectory.append(a, r, policy, value, nx)
    if done:
      break
  memory.store(trajectory)


def train(env, muzero_net, memory):
  actors = ThreadActors(ACTORS, play, (env, muzero_net, memory))
  actors.run()
  if memory.size() >= BATCH_SIZE:
    trajectories = memory.sample(BATCH_SIZE)
    muzero_net.train(trajectories)


if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  muzero_net = MuZeroNet()
  memory = Memory()
  try:
    for epoch in range(EPOCHS):
      print('Epoch ' + epoch)
      train(env, muzero_net, memory)
  finally:
    env.close()
