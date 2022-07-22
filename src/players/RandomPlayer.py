from src.players.ConnectFourPlayer import ConnectFourPlayer
import random

class RandomPlayer(ConnectFourPlayer):
  gaussian_col_weights = [0.3, 0.6, 0.8, 1, 0.8, 0.6, 0.3]
                          
  def __init__(self, gaussian=True):
    super(RandomPlayer, self).__init__()
    self.gaussian = gaussian

  def pick_move(self, game):
    valid_cols = game.valid_moves()
    if (self.gaussian):
      return self.pick_random_gaussian(valid_cols)
    else:
      return self.pick_random_uniform(valid_cols)

  @classmethod
  def pick_random_uniform(cls, cols):
    return random.choice(valid_cols)

  @classmethod
  def pick_random_gaussian(cls, cols):
    valid_col_weights = [cls.gaussian_col_weights[col] for col in cols]
    return random.choices(cols, weights = valid_col_weights, k = 1)[0]