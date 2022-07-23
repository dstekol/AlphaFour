from src.players.ConnectFourPlayer import ConnectFourPlayer
from src.players.RandomPlayer import RandomPlayer
from src.players.components.MCTS import MCTS

class MCTSPlayer(ConnectFourPlayer):
  def __init__(self, gaussian=True, c_puct=1, mcts_iters=4000, temperature=0):
    super(MCTSPlayer, self).__init__()
    move_prob_func = RandomPlayer.pick_random_gaussian if gaussian \
      else RandomPlayer.pick_random_uniform
    eval_func = lambda game, actions: (move_prob_func(actions), None)
    self.mcts = MCTS(eval_func=eval_func, mcts_iters=mcts_iters, explore_coeff=c_puct, temperature=temperature)

  def pick_move(self, game):
    return self.mcts.search(game)
