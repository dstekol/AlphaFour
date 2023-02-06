import random

class RandomPlayer:
  """
  Player which selects moves randomly, with either a uniform or gaussian (center-biased) probability distribution.
  """

  # approximate gaussian weight distribution for biasing column selection toward center
  gaussian_col_weights = [0.3, 0.6, 0.8, 1, 0.8, 0.6, 0.3]
                          
  def __init__(self, gaussian=True):
    super(RandomPlayer, self).__init__()
    self.gaussian = gaussian

  def pick_move(self, game):
    """
    Selects random column according to either uniform or gaussian policy.
    """
    valid_cols = game.valid_moves()
    if (self.gaussian):
      return self.pick_random_gaussian(valid_cols)
    else:
      return self.pick_random_uniform(valid_cols)

  @classmethod
  def pick_random_uniform(cls, cols):
    """
    Returns:
    Integer: random column according to uniform weighting
    """
    return random.choice(valid_cols)

  @classmethod
  def pick_random_gaussian(cls, cols):
    """
    Returns:
    Integer: random column according to center-biased weighting
    """
    valid_col_weights = [cls.gaussian_col_weights[col] for col in cols]
    return random.choices(cols, weights = valid_col_weights, k = 1)[0]