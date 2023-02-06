from src.players.RandomPlayer import RandomPlayer
from src.players.mcts.MCTS import MCTS

class MCTSPlayer:
  """
  Uses Monte Carlo Tree Search algorithm (variant proposed by AlphaZero paper).
  At terminal nodes, uses random policy (either uniform or gaussian) in place of trained policy network.
  """

  def __init__(self, mcts_args, gaussian=True):
    """
    Args:
    mcts_args (Dict[String, Any]): dictionary containing arguments for MCTS object instantiation.
      Must contain: mcts_iters, discount, explore_coeff, temperature, dirichlet_coeff, dirichlet_alpha, num_threads
    gaussian (Boolean): whether to use gaussian/center-biased policy when terminal node reached. If False, will use uniform policy instead.
    """
    super(MCTSPlayer, self).__init__()
    action_distribution = RandomPlayer.gaussian_col_weights if gaussian \
      else [1] * len(RandomPlayer.gaussian_col_weights)
    eval_func = lambda game: (action_distribution, None)
    self.mcts = MCTS(eval_func=eval_func, **mcts_args)

  def pick_move(self, game):
    return self.mcts.search(game)
