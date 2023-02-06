import math
import random
import numpy as np
import threading

def hashable_game_state(game):
  """
  Helper function for getting bytestring representation of ConnectFour object, which can be used as hashable dict key

  Args:
  game (ConnectFour): the game object to convert
  """
  return game.board.tobytes()

class MCTS:
  """
  Performs Monte Carlo Tree Search
  """

  def __init__(self, eval_func, 
                     mcts_iters, 
                     discount,
                     explore_coeff, 
                     temperature, 
                     dirichlet_coeff, 
                     dirichlet_alpha,
                     num_threads):
    """
    Args:
    eval_func (Callable): evaluation function with signature: 
      game (ConnectFour object) -> action_vals (np.ndarray of shape (7,) ), state_val (np.ndarray of shape (1,) )
    mcts_iters (Integer): number of MCTS trajectory rollouts to perform
    discount (Float): discount to apply to reward for each timestep. Float in range [0, 1]
    explore_coeff (Float): strength of exploration term when selecting next node to expand
    temperature (Float): temperature term when selecting final move.
      Set to 0 to deterministically select max-value move, or 1 to select stochastically 
      with probabilities proportional to visit counts.
    dirichlet_coeff (Float): strength of dirichlet noise (added for additional exploration).
      Float in range [0,1]. Set to 0 for no noise, or 1 for only noise.
    num_threads (Integer): number of threads to use for simultaneous exploration.
    """

    # maps game states to MCTSNode objects representing computed MCTS stats (or integer game outcome for terminal nodes, or threading.Condition for pending computations)
    self.nodes = dict()

    # synchronization object for modifying nodes dict
    self.nodes_lock = threading.RLock()

    # MCTS rollout counter
    self.global_sims_finished = 0

    # lock for global_sims_finished counter
    self.global_counter_lock = threading.RLock()

    # MCTS params
    self.explore_coeff = explore_coeff
    self.eval_func = eval_func
    self.temperature = temperature
    self.dirichlet_coeff = dirichlet_coeff
    self.dirichlet_alpha = dirichlet_alpha
    self.discount = discount
    self.mcts_iters = mcts_iters
    self.num_threads = num_threads


  def update_temperature(self, temperature):
    """
    Updates temperature term for selecting final move. 
      Set to 0 to deterministically select max-value move, or 1 to select stochastically 
      with probabilities proportional to visit counts.

    Args:
    temperature: Float in range [0, 1]. High temperature smooths values, low temperature biases toward max. When temperature is 0, max is selected deterministically.
    """
    self.temperature = temperature

  def current_action_scores(self, game, temperature=None):
    """
    Gets the computed MCTS values of all actions for the given game.

    Args:
    game: ConnectFour object representing game state
    temperature: Float in range [0, 1]. High temperature smooths values, low temperature biases toward max. When temperature is 0, max is selected deterministically.
      If None, defaults to self.temperature

    Returns:
    Computed MCTS values of all actions for the given game.
    """

    if (temperature is None):
      temperature = self.temperature
    node = self.nodes[hashable_game_state(game)]
    actions = list(range(game.cols))
    return node.action_scores(actions, temperature)

  def search(self, game):
    """
    Performs PUCT/MCTS search (as described in AlphaZero paper) to find best current move.
    Multithreading is used to avoid GPU bottleneck.

    Args:
    game (ConnectFour): game object representing game state.
    
    Returns:
    Integer: Best current move, selected nondeterministically with probabilities proportional to temperature-scaled exponentiated visit counts.
    """

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
    """
    Replaces condition object (used as placeholder) in node dict with MCTSNode object

    Args:
    state (bytes): bytestring representing current state, obtained via hashable_game_state()
    node (MCTSNode): node containing computed stats for current state
    """

    condition = self.nodes[state]
    if (not isinstance(condition, threading.Condition)):
      raise LookupError("Overwriting precomputed node")
    self.nodes[state] = node
    with condition:
      condition.notify_all()

  def search_thread(self, game):
    """
    Runs a single thread of PUCT/MCTS search until desired number of simulated rollouts is reached.
    
    Args:
    game (ConnectFour): game object representing game state
    """

    orig_game = game
    num_sims_finished = 0
    while (num_sims_finished < self.mcts_iters):
      game = orig_game.copy()
      is_over = False
      trajectory = []
      while(not is_over):
        state = hashable_game_state(game)

        with self.nodes_lock:
          state_present = state in self.nodes
          # put sync condition into node dict as placeholder
          if (not state_present):
            self.nodes[state] = threading.Condition()

        if (not state_present):
          is_over, outcome = game.game_over()

          # if game ended, use actual outcome (put integer into node dict rather than MCTSNode)
          if (is_over):
            self._replace_condition_with_node(state, outcome)
            break

          # otherwise use predicted outcome (insert MCTSNode into node dict)
          actions = game.valid_moves()
          priors, outcome = self.eval_func(game)
          node = MCTSNode(actions, priors, self.explore_coeff, game.cols)
          self._replace_condition_with_node(state, node)
          if (outcome is not None):
            break
        else:
          item = self.nodes[state]
          if (isinstance(item, threading.Condition)): # pending computation
            with item:
              if (isinstance(self.nodes[state], threading.Condition)):
                item.wait()
              node = self.nodes[state]
          elif (isinstance(item, int)): # terminal node
            outcome = item
            break
          elif (isinstance(item, MCTSNode)): # nonterminal node
            node = item
          else:
            raise LookupError("Illegal object in node map")
        action = node.max_uct_action()
        node.incur_virtual_loss(action) # discourage other threads from taking same path
        trajectory.append((state, action))
        game.perform_move(action)

      # backpropagate value of terminal node to ancestor nodes
      for j, (state, action) in enumerate(trajectory):
        signed_outcome = outcome * orig_game.player * (-1 if j % 2 == 1 else 1)
        moves_until_end = len(trajectory) - j - 1
        discounted_outcome = signed_outcome * (self.discount ** moves_until_end)
        self.nodes[state].update_action_value(action, discounted_outcome)  # FIXED from signed_outcome
      
      # update rollout counter
      with self.global_counter_lock:
        self.global_sims_finished += 1
        num_sims_finished = self.global_sims_finished

    


class MCTSNode:
  """
  Threadsafe helper class for representing nodes of MCTS search. Stores rollout statistics and computes UCT scores.
  """

  def __init__(self, actions, priors, explore_coeff, total_actions):
    """
    Args:
    actions (List[Integer]): list of valid game actions
    priors (List[Float]): list of action priors (ex. scores from policy head)
    explore_coeff (Float): exploration coefficient (higher values result in more exploration relative to exploitation)
    total_actions (Integer): total number of possible actions
    """

    self.priors = priors
    self.actions = actions
    self.explore_coeff = explore_coeff
    self.action_count = np.zeros((total_actions,))
    self.total_count = 0
    self.action_total_value = np.zeros((total_actions,))
    self.lock = threading.RLock()

  def q_value(self, action):
    """
    Computes average action value (q-value)

    Args:
    action (Integer): action for which corresponding q-value is desired

    Returns:
    Action q-value (Float)
    """
    with self.lock:
      return self.action_total_value[action] / (self.action_count[action] + 1e-6)

  def action_uct(self, action):
    """
    Gets UCT score of specified action

    Args:
    action (Integer)

    Returns:
    UCT score of action (Float)
    """
    with self.lock:
      conf_bound = self.priors[action] * math.sqrt(self.total_count) / (1 + self.action_count[action])
      return self.q_value(action) + self.explore_coeff * conf_bound

  def max_uct_action(self):
    """
    Returns action (Integer) with maximum UCT value. 
    If multiple actions have same UCT value, returns one of max-value actions uniformly at random.
    """

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
    """
    Gets scores of specified actions.

    Args:
    actions (List[Integer]): actions whose scores to return
    temperature (Float): controls the probability weighting of actions: the higher the temperature, the closer to uniform the weighting distribution becomes. 
      If set to 0, the max-score action is chosen deterministically.

    Returns:
    List[Float]: list of corresponding action scores (temperature-exponentiated visit counts)
    """

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
    """
    Lowers the score of the specified action in order to discourage other threads from following a previously explored path.
    This adjustment is undone in update_action_value() once the outcome value has been computed and propagated to ancestor nodes.

    Args:
    action (Integer)
    """

    with self.lock:
      self.total_count += 1
      self.action_count[action] += 1
      self.action_total_value[action] -= 1

  def sample_best_action(self, temperature, dirichlet_coeff, dirichlet_alpha):
    """
    Stochastically selects an action with probabilisted proportional to temperature-exponentiated visit counts (with optional dirichlet noise).

    Args:
    temperature (Float): Controls stochastisity. Set to 0 to deterministically select max-value move, or 1 to select stochastically 
      with probabilities proportional to visit counts.
    dirichlet_coeff (Float): Float in range [0,1] controlling strength of dirichlet noise (0 means no noise, 1 means all noise).
    dirichlet_alpha (Float): dirichlet distribution alpha parameter

    Returns:
    Integer: sampled action
    """

    with self.lock:
      weights = self.action_scores(self.actions, temperature)
      noise = np.random.dirichlet(alpha = [dirichlet_alpha] * len(weights))
      noisy_weights = (1 - dirichlet_coeff) * weights + dirichlet_coeff * noise
      move = random.choices(self.actions, weights=noisy_weights, k=1)[0]
      return move

  def update_action_value(self, action, value):
    """
    Updates running average of action value. Also undoes virtual loss.

    Args:
    action (Integer): The action whose value to update
    value (Float): Most recently sampled value for this action
    """

    with self.lock:
      self.action_total_value[action] += value + 1






