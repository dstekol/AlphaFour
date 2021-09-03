from src.players.ConnectFourPlayer import ConnectFourPlayer
import random

class RandomPlayer(ConnectFourPlayer):
  gaussian_col_weights = [0.3, 0.6, 0.8, 1, 0.8, 0.6, 0.3]
                          
  def __init__(self, gaussian=True):
    super(RandomPlayer, self).__init__(game)
    self.gaussian = gaussian

  def pick_move(self, game):
    valid_cols = [col for col in range(game.cols) if game.level[col] >= 0]
    if (self.gaussian):
      return self.pick_random_gaussian(valid_cols)
    else:
      return random.choice(valid_cols)

  @classmethod
  def pick_random_gaussian(cls, cols):
    valid_col_weights = [cls.gaussian_col_weights[col] for col in cols]
    return random.choices(cols, weights = valid_col_weights, k = 1)[0]