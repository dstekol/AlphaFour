import math
import random
import numpy as np
import threading

def hashable_game_state(game):
  return game.board.tobytes()

class MCTS:
  def __init__(self, eval_func, 
                     mcts_iters, 
                     discount,
                     explore_coeff, 
                     temperature, 
                     dirichlet_coeff, 
                     dirichlet_alpha,
                     num_threads):
    self.nodes = dict()
    self.explore_coeff = explore_coeff
    self.eval_func = eval_func
    self.temperature = temperature
    self.dirichlet_coeff = dirichlet_coeff
    self.dirichlet_alpha = dirichlet_alpha
    self.discount = discount
    self.mcts_iters = mcts_iters
    self.num_threads = num_threads

  def update_temperature(self, temperature):
    self.temperature = temperature

  def current_action_scores(self, game, temperature=None):
    if (temperature is None):
      temperature = self.temperature
    node = self.nodes[hashable_game_state(game)]
    actions = list(range(game.cols))
    return node.action_scores(actions, temperature)

  def search(self, game):
    self.global_sims_finished = 0
    lock = threading.RLock()
    threads = []
    for i in range(self.num_threads):
      t = threading.Thread(target=self.search_thread, args=(game, lock))
      threads.append(t)
    [t.start() for t in threads]
    [t.join() for t in threads]
    orig_state = hashable_game_state(game)
    current_node = self.nodes[orig_state]
    return current_node.sample_best_action(self.temperature, 
                                           self.dirichlet_coeff, 
                                           self.dirichlet_alpha)


  def search_thread(self, game, lock):
    orig_game = game
    num_sims_finished = 0
    while (num_sims_finished < self.mcts_iters):
      #print(num_sims_finished)
      game = orig_game.copy()
      is_over = False
      trajectory = []
      while(not is_over):
        state = hashable_game_state(game)
        if (state not in self.nodes):
          is_over, outcome = game.game_over()
          if (is_over):
            break
          actions = game.valid_moves()
          priors, outcome = self.eval_func(game, actions)
          node = MCTSNode(actions, priors, self.explore_coeff, game.cols)
          self.nodes[state] = node
          if (outcome is not None):
            break
        else:
          node = self.nodes[state]
        action = node.max_uct_action()
        node.incur_virtual_loss(action)
        trajectory.append((state, action))
        game.perform_move(action)
      for j, (state, action) in enumerate(trajectory):
        signed_outcome = outcome * orig_game.player * (-1 if j % 2 == 1 else 1)
        moves_until_end = len(trajectory) - j - 1
        discounted_outcome = signed_outcome * (self.discount ** moves_until_end)
        self.nodes[state].update_action_value(action, signed_outcome)
      with lock:
        self.global_sims_finished += 1
        num_sims_finished = self.global_sims_finished
    #print(f"exiting {threading.get_ident()}__________________________")

    


class MCTSNode:
  def __init__(self, actions, priors, explore_coeff, total_actions):
    self.priors = priors
    self.actions = actions
    self.explore_coeff = explore_coeff
    self.action_count = np.zeros((total_actions,))
    self.total_count = 0
    self.action_total_value = np.zeros((total_actions,))
    self.lock = threading.RLock()

  def q_value(self, action):
    with self.lock:
      return float(self.action_total_value[action]) / (float(self.action_count[action]) + 1e-6)

  def action_uct(self, action):
    with self.lock:
      conf_bound = math.sqrt(self.total_count) / (1 + self.action_count[action])
      return self.q_value(action) + self.explore_coeff * conf_bound

  def max_uct_action(self):
    with self.lock:
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

  def action_scores(self, actions, temperature):
    with self.lock:
      if (temperature == 0):
        max_count = self.action_count.max()
        weights = (self.action_count == max_count)[actions]
      else:
        weights = np.zeros_like(actions)
        for i, action in enumerate(actions):
          weights[i] = self.action_count[action]**(1 / temperature)
      weights = weights / weights.sum()
      return weights

  def incur_virtual_loss(self, action):
    with self.lock:
      self.total_count += 1
      self.action_count[action] += 1
      self.action_total_value[action] -= 1

  def sample_best_action(self, temperature, dirichlet_coeff, dirichlet_alpha):
    with self.lock:
      weights = self.action_scores(self.actions, temperature)
      noise = np.random.dirichlet(alpha = [dirichlet_alpha] * len(weights))
      weights = (1 - dirichlet_coeff) * weights + dirichlet_coeff * noise
      return random.choices(self.actions, weights=weights, k=1)[0]

  def update_action_value(self, action, value):
    with self.lock:
      #self.total_count += 1
      self.action_total_value[action] += value + 1
      #self.action_count[action] += 1







