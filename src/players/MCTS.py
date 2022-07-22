import math
import random
import numpy as np

class MCTSNode:
  def __init__(self, actions, priors, c_puct, total_actions):
    self.priors = priors
    self.actions = actions
    self.c_puct = c_puct
    self.action_count = [0] * total_actions #{action: 0 for action in actions}
    self.total_count = 0
    self.action_total_value = [0] * total_actions # {action: 0 for action in actions}

  def q_value(self, action):
    return float(self.action_total_value[action]) / (float(self.action_count[action]) + 1e-6)

  def action_uct(self, action):
    conf_bound = math.sqrt(self.total_count) / (1 + self.action_count[action])
    return self.q_value(action) + self.c_puct * conf_bound

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
    #metric_func = lambda action: self.action_uct(action)
    #return self._max_action_by_metric(metric_func)

  def get_max_count_action(self):
    metric_func = lambda action: self.action_count[action]
    return self._max_action_by_metric(metric_func)

  def _max_action_by_metric(self, metric_func):
    action_metrics = map(metric_func, self.actions)
    max_metric = max(action_metrics)
    max_metric_actions = list(filter(lambda action: metric_func(action) == max_metric, self.actions))
    return random.choice(max_metric_actions)

  def get_exp_action_scores(self, actions, temperature):
    weights = np.zeros((len(actions),))
    for i, action in enumerate(actions):
      #if (action in self.action_count):
      weights[i] = self.action_count[action]**(1) # TODO insert temperature
    weights = weights / weights.sum()
    return weights

  def sample_best_action(self, temperature, dirichlet_coeff, dirichlet_alpha):
    weights = self.get_exp_action_scores(self.actions, temperature)
    weights += dirichlet_coeff * np.random.dirichlet(alpha = [dirichlet_alpha] * len(weights)) # TO
    weights = weights / weights.sum()
    #return random.choices(self.actions, weights=weights, k=1)[0] #TODO
    return self.actions[weights.argmax()]

  def update_action_value(self, action, value):
    self.total_count += 1
    self.action_total_value[action] += value
    self.action_count[action] += 1


def extract_game_state_tuple(game):
  #return tuple(map(tuple, game.board))
  return game.board.tobytes()

class MCTS:
  def __init__(self, eval_func, mcts_iters, c_puct=1, temperature=1e-5, resign_threshold=-0.85, dirichlet_coeff=0, dirichlet_alpha=0.03):
    self.nodes = dict()
    self.c_puct = c_puct
    self.eval_func = eval_func
    self.temperature = temperature # TODO
    self.resign_threshold = resign_threshold
    self.dirichlet_coeff = dirichlet_coeff
    self.dirichlet_alpha = dirichlet_alpha
    self.mcts_iters = mcts_iters

  def update_temperature(self, temperature):
    self.temperature = temperature

  def current_action_scores(self, game):
    node = self.nodes[extract_game_state_tuple(game)]
    actions = list(range(game.cols))
    return node.get_exp_action_scores(actions, temperature=1e-5) # TODO

  def search(self, game, mcts_iters=None):
    if (mcts_iters is None):
      mcts_iters = self.mcts_iters
    orig_game = game
    for i in range(mcts_iters):
      game = orig_game.copy()
      is_over = False
      trajectory = []
      while(not is_over):
        state = extract_game_state_tuple(game)
        if (state not in self.nodes):
          actions = game.valid_moves()
          priors, outcome = self.eval_func(game, actions)
          if (outcome is not None):
            break
          else:
            node = MCTSNode(actions, priors, self.c_puct, game.cols)
            self.nodes[state] = node
        node = self.nodes[state]
        action = node.get_max_uct_action()
        trajectory.append((state, action))
        game.perform_move(action)
        is_over, outcome = game.game_over()
      for i, (state, action) in enumerate(trajectory):
        signed_outcome = outcome * orig_game.player * (-1 if i % 2 == 1 else 1)
        self.nodes[state].update_action_value(action, signed_outcome)
    orig_state = extract_game_state_tuple(orig_game)
    #max_count_move = self.nodes[orig_state].get_max_count_action()
    current_node = self.nodes[orig_state]
    return current_node.sample_best_action(self.temperature, self.dirichlet_coeff, self.dirichlet_alpha)







