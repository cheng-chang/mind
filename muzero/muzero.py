import threading
import math
import typing

from tqdm import tqdm
import gym
import tensorflow as tf
import numpy as np


FLOAT_MAX = float('inf')
EPOCHS = int(1e6)
EPOCH_STEPS = 100
PLAN_STEPS = 80
ACTORS = 10
BATCH_SIZE = 64
ACTIONS = 2
ROOT_PRIOR_NOISE_DIRICHLET_ALPHA = 0.03
ROOT_PRIOR_NOISE_FRACTION = 0.25
UPPER_CONFIDENCE_BOUND_C1 = 1.25
UPPER_CONFIDENCE_BOUND_C2 = 19652
DISCOUNT = 0.997


class RepresentationNet:
  pass


class DynamicsNet:
  pass


class PredictionNet:
  pass


class MuZeroNet:
  def represent(self, raw_state) -> MuZeroNetOutput:
    pass

  def transit(self, state, action) -> MuZeroNetOutput:
    pass

  def train(self, trajectories):
    pass

  def training_steps(self):
    pass


class MuZeroNetOutput(typing.NamedTuple):
  state: typing.List[float]
  reward: float
  policy: typing.Dict[int, float]
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
    prior = self._compute_action_prior(netoutput.policy, parent is None)
    self._action_stats = {a: ActionStats(prior[a]) for a in ACTIONS}
    self._children = {}

  @staticmethod
  def _compute_action_prior(policy, should_add_noise):
    policy = {a: math.exp(policy[a]) for a in ACTIONS}
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
    visit_counts = sum(map(lambda a: stats[a].visit_count(), stats))
    visit_count = stat.visit_count()
    c1 = UPPER_CONFIDENCE_BOUND_C1
    c2 = UPPER_CONFIDENCE_BOUND_C2

    prior_weight = math.sqrt(visit_counts) / (1 + visit_count)
    prior_weight *= (c1 + math.log((visit_counts + c2 + 1) / c2))
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

  def visit_counts(self):
    return {action: stat.visit_count() for (action, stat) in self._action_stats.items()}


class MonteCarloTreeSearch(Planner):
  def __init__(self, muzero: MuZeroNet):
    self._muzero = muzero

  def plan(self, state):
    output = self._muzero.represent(state)
    root = Node(output)
    for _ in range(PLAN_STEPS):
      self._plan(root)
    return self._choose_action(root)

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
    visit_counts = root.visit_counts()
    actions = visit_counts.keys()
    counts = np.array([visit_counts[a] for a in actions])
    t = self._visit_count_temperature()
    idx = self._sample_index(counts, t)
    return actions[idx]

  def _visit_count_temperature(self):
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
  """A trajectory is (x0, a0, r0, x1, a1, r1, ..., xn).

  x is the raw observation from the environment,
  a is action,
  r is the immediate reward of the previous action.

  NOTE: x will be transformed to a state representation in the hidden space,
  the transformed state uses symbol s.
  """
  def __init__(self, x0):
    self._states = [x0]
    self._actions = []
    self._rewards = []

  def current_state(self):
    return self._states[-1]

  def append(self, current_action, reward, next_state):
    self._actions.append(current_action)
    self._rewards.append(reward)
    self._states.append(next_state)


class Memory:
  def store(self, trajectory):
    """Store trajectory.

    Might be called concurrently by multiple actors, need to be thread-safe.
    """
    pass

  def size(self):
    pass

  def sample(self, batch_size):
    assert(self.size() >= batch_size)
    pass


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


def play(env, planner, memory):
  """Reset the environment and collect a new trajectory into memory."""
  trajectory = Trajectory(env.reset())
  for _ in tqdm(range(EPOCH_STEPS)):
    x = trajectory.current_state()
    a = planner.plan(x)
    nx, r, done, _ = env.step(a)
    trajectory.append(a, r, nx)
    if done:
      break
  memory.store(trajectory)


def train(env, muzero, planner, memory):
  actors = ThreadActors(ACTORS, play, (env, planner, memory))
  actors.run()
  if memory.size() >= BATCH_SIZE:
    trajectories = memory.sample(BATCH_SIZE)
    muzero.train(trajectories)


if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  muzero = MuZeroNet()
  memory = Memory()
  planner = MonteCarloTreeSearch()
  try:
    for epoch in range(EPOCHS):
      print('Epoch ' + epoch)
      train(env, muzero, planner, memory)
  finally:
    env.close()
