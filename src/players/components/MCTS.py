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
    #self.nodes_orig = dict()
    self.nodes_lock = threading.RLock()
    self.global_sims_finished = 0
    self.global_counter_lock = threading.RLock()
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
    threads = []
    for i in range(self.num_threads):
      t = threading.Thread(target=self.search_thread, args=(game,))
      threads.append(t)
    [t.start() for t in threads]
    [t.join() for t in threads]
    orig_state = hashable_game_state(game)
    current_node = self.nodes[orig_state]
    return current_node.sample_best_action(self.temperature, 
                                           self.dirichlet_coeff, 
                                           self.dirichlet_alpha)

  def _replace_condition_with_node(self, state, node):
    condition = self.nodes[state]
    if (not isinstance(condition, threading.Condition)):
      raise LookupError("Overwriting precomputed node")
    self.nodes[state] = node
    with condition:
      #print(f"finishing adding state: {threading.get_ident()}")
      condition.notify_all()

  #def print_entropy(self):
  #  s = 0
  #  c = 0
  #  for state in self.nodes:
  #    node = self.nodes[state]
  #    if (not isinstance(node, MCTSNode)):
  #      continue
      
  #    s += node.entropy()
  #    c += 1
  #  print("avg entropy")
  #  if (c != 0):
  #    print(s / c)

  def search_thread(self, game):
    orig_game = game
    num_sims_finished = 0
    while (num_sims_finished < self.mcts_iters):
      #print(f"thread: {threading.get_ident()}")
      #print(num_sims_finished)
      game = orig_game.copy()
      is_over = False
      trajectory = []
      while(not is_over):
        state = hashable_game_state(game)

        with self.nodes_lock:
          state_present = state in self.nodes
          if (not state_present):
            #print(f"adding state: {threading.get_ident()}")
            self.nodes[state] = threading.Condition()
            #self.nodes_orig[state] = threading.get_ident()

        if (not state_present):
          is_over, outcome = game.game_over()
          if (is_over):
            self._replace_condition_with_node(state, outcome)
            break
          actions = game.valid_moves()
          priors, outcome = self.eval_func(game, actions)
          node = MCTSNode(actions, priors, self.explore_coeff, game.cols)
          self._replace_condition_with_node(state, node)
          if (outcome is not None):
            break
        else:
          item = self.nodes[state]
          if (isinstance(item, threading.Condition)):
            with item:
              #orig_t = self.nodes_orig[state]
              #print(f"waiting for state: {threading.get_ident()} added by {orig_t}")
              if (isinstance(self.nodes[state], threading.Condition)):
                item.wait()
              #print(f"restarting thread: {threading.get_ident()} added by {orig_t}")
              node = self.nodes[state]
          elif (isinstance(item, int)):
            outcome = item
            break
          elif (isinstance(item, MCTSNode)):
            node = item
          else:
            raise LookupError("Illegal object in node map")
        action = node.max_uct_action()
        node.incur_virtual_loss(action)
        trajectory.append((state, action))
        game.perform_move(action)
      for j, (state, action) in enumerate(trajectory):
        signed_outcome = outcome * orig_game.player * (-1 if j % 2 == 1 else 1)
        moves_until_end = len(trajectory) - j - 1
        discounted_outcome = signed_outcome * (self.discount ** moves_until_end)
        self.nodes[state].update_action_value(action, signed_outcome)
      with self.global_counter_lock:
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

  #def entropy(self):
  #  if (self.total_count == 0):
  #    return 0
  #  norm_act_counts = (self.action_count + 0.01) / self.action_count.sum()
  #  return -(norm_act_counts * np.log(norm_act_counts)).sum()

  def action_scores(self, actions, temperature):
    with self.lock:
      if (self.total_count == 0):
        raise ValueError("Weights sum to 0")
      if (temperature == 0):
        max_count = self.action_count.max()
        weights = (self.action_count == max_count)[actions].astype(int)
      else:
        weights = self.action_count[actions]
        weights = weights / weights.sum()
        weights = weights ** (1 / temperature)
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
      noisy_weights = (1 - dirichlet_coeff) * weights + dirichlet_coeff * noise
      move = random.choices(self.actions, weights=noisy_weights, k=1)[0]
      #if (move != self.actions[weights.argmax()]):
      #  print(weights.round(4))
      #  print((self.action_count[self.actions] / self.action_count[self.actions].sum()).round(4))
      #  #print(noisy_weights.round(4))
      return move

  def update_action_value(self, action, value):
    with self.lock:
      #self.total_count += 1
      self.action_total_value[action] += value + 1
      #self.action_count[action] += 1







