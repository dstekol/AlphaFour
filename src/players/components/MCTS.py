import math
import random
import numpy as np

def hashable_game_state(game):
  return game.board.tobytes()

class MCTS:
  def __init__(self, eval_func, 
                     mcts_iters, 
                     explore_coeff=1, 
                     temperature=0, 
                     dirichlet_coeff=0, 
                     dirichlet_alpha=0.03):
    self.nodes = dict()
    self.explore_coeff = explore_coeff
    self.eval_func = eval_func
    self.temperature = temperature
    self.dirichlet_coeff = dirichlet_coeff
    self.dirichlet_alpha = dirichlet_alpha
    self.mcts_iters = mcts_iters

  def update_temperature(self, temperature):
    self.temperature = temperature

  def current_action_scores(self, game):
    node = self.nodes[hashable_game_state(game)]
    actions = list(range(game.cols))
    return node.get_exp_action_scores(actions, self.temperature)

  def search(self, game, mcts_iters=None):
    if (mcts_iters is None):
      mcts_iters = self.mcts_iters
    orig_game = game
    for i in range(mcts_iters):
      game = orig_game.copy()
      is_over = False
      trajectory = []
      while(not is_over):
        state = hashable_game_state(game)
        if (state not in self.nodes):
          actions = game.valid_moves()
          priors, outcome = self.eval_func(game, actions)
          node = MCTSNode(actions, priors, self.explore_coeff, game.cols)
          self.nodes[state] = node
          if (outcome is not None):
            break
        node = self.nodes[state]
        action = node.get_max_uct_action()
        trajectory.append((state, action))
        game.perform_move(action)
        is_over, outcome = game.game_over()
      for i, (state, action) in enumerate(trajectory):
        signed_outcome = outcome * orig_game.player * (-1 if i % 2 == 1 else 1)
        self.nodes[state].update_action_value(action, signed_outcome)
    orig_state = hashable_game_state(orig_game)
    current_node = self.nodes[orig_state]
    return current_node.sample_best_action(self.temperature, 
                                           self.dirichlet_coeff, 
                                           self.dirichlet_alpha)


class MCTSNode:
  def __init__(self, actions, priors, explore_coeff, total_actions):
    self.priors = priors
    self.actions = actions
    self.explore_coeff = explore_coeff
    self.action_count = np.zeros((total_actions,))
    self.total_count = 0
    self.action_total_value = np.zeros((total_actions,))

  def q_value(self, action):
    return float(self.action_total_value[action]) / (float(self.action_count[action]) + 1e-6)

  def action_uct(self, action):
    conf_bound = math.sqrt(self.total_count) / (1 + self.action_count[action])
    return self.q_value(action) + self.explore_coeff * conf_bound

  def get_max_uct_action(self):
    max_action_uct = -math.inf
    max_actions = []
    for action in self.actions:
      action_uct = self.action_uct(action)
      if (action_uct > max_action_uct):
        max_actions = [action]
        max_action_uct = action_uct
      elif (action_uct == max_action_uct):
        max_actions.append(action)
    if (len(max_actions) == 1):
      return max_actions[0]
    else:
      return random.choice(max_actions)

  def get_exp_action_scores(self, actions, temperature):
    if (temperature == 0):
      max_count = self.action_count.max()
      weights = (self.action_count == max_count)[actions]
    else:
      weights = np.zeros_like(actions)
      for i, action in enumerate(actions):
        weights[i] = self.action_count[action]**(1 / temperature)
    weights = weights / weights.sum()
    return weights

  def sample_best_action(self, temperature, dirichlet_coeff, dirichlet_alpha):
    weights = self.get_exp_action_scores(self.actions, temperature)
    noise = np.random.dirichlet(alpha = [dirichlet_alpha] * len(weights))
    weights = (1 - dirichlet_coeff) * weights + dirichlet_coeff * noise
    return random.choices(self.actions, weights=weights, k=1)[0]

  def update_action_value(self, action, value):
    self.total_count += 1
    self.action_total_value[action] += value
    self.action_count[action] += 1







