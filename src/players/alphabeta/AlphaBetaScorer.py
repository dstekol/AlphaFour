import numpy as np

class AlphaBetaScorer:
  """
  Helper class implementing handcrafted evaluation function for Connect Four
  """

  def __init__(self,
               discount,
               win_reward, 
               low_threat_reward_bounds, 
               high_threat_reward_bounds, 
               stacked_threat_reward_bounds):
    """
    Args:
    discount (Float): factor by which final reward is multiplied for each step. 
      Setting discount < 1 (ex. 0.96) encourages winning quickly, since the reward becomes smaller for victories further in the future
    win_reward (Float): reward for winning
    low_threat_reward_bounds (Tuple[Float, Float]): reward bounds tuple of form (min, max) for a "low" threat (see is_high_reward() for explanation).
      Agent will receive min reward for the first such threat, and will asymptotically approach max reward for additional threats.
    high_threat_reward_bounds (Tuple[Float, Float]): reward bounds tuple of form (min, max) for a "high" threat (see is_high_reward() for explanation)
      Agent will receive min reward for the first such threat, and will asymptotically approach max reward for additional threats.
    stacked_threat_reward_bounds (Tuple[Float, Float]): reward for threats stacked on top of each other
      Agent will receive min reward for the first such threat, and will asymptotically approach max reward for additional threats.
    """
    self.discount = discount
    self.win_reward = win_reward
    self.low_threat_reward_bounds = low_threat_reward_bounds
    self.high_threat_reward_bounds = high_threat_reward_bounds
    self.stacked_threat_reward_bounds = stacked_threat_reward_bounds

  def score(self, game, winner, depth):
    """
    Returns the depth-discounted evaluation score of the given game. 
    Positive when player 1 winning, negative when player 2 winning.
    Score = 1 means player 1 guaranteed to win; Score = -1 means player 2 guaranteed to win.

    Args:
    game (ConnectFour): the game to evaluate
    winner (Integer): 1 if player 1 wins, -1 if player 2 wins, 0 if tie
    depth (Integer): number of moves until game ends

    Returns:
    Float: score
    """
    return self.base_score(game, winner)  * (self.discount ** depth)

  def base_score(self, game, winner):
    """
    Computes undiscounted evaluation score for given game.

    Args:
    game (ConnectFourWithThreats): game to evaluate
    winner (Integer): 1 if player 1 has won, -1 if player 2 has won, 0 for tie or if game being evaluated is unfinished.

    Returns:
    Float: score
    """
    if (winner != 0):
      return winner * self.win_reward
    
    reward = 0
    for player in [-1, 1]:
      # count number of high and low threats
      num_low_threats = 0
      num_high_threats = 0
      for threat in game.threats[player]:
        if (self.is_high_reward(threat)):
          num_high_threats += 1
        else:
          num_low_threats += 1

      # compute threat rewards via asymptotic mapping 
      # (player gets min reward for having at least one threat, 
      # and asymptotically approaches max reward as number of threats increases)
      reward += player * self.map_to_asymptotic(num_low_threats, 
                                                self.low_threat_reward_bounds)
      reward += player * self.map_to_asymptotic(num_high_threats,
                                                self.high_threat_reward_bounds)
    
    stacked_threat_diff = self.count_stacked_threat_diff(game)
    player = 1 if stacked_threat_diff > 0 else -1
    reward += player * self.map_to_asymptotic(abs(stacked_threat_diff), 
                                                self.stacked_threat_reward_bounds)
    return reward

  def count_stacked_threat_diff(self, game):
    """
    Computes the number of stacked threats each player has.
    A stacked threat occurs when the open spot of one threat occurs directly above the open spot of another threat.
    Such configurations often guarantee a win for the owner of the stacked threats.
    Rather than returning separate values for each player, this function returns the difference between the number of stacked threats of player 1 and player 2.
    Therefore, a positive value means player 1 has more stacked threats, a negative value means player 2 has more stacked threats, and 0 means both players have an equal number of stacked threats.

    Args:
    game (ConnectFourWithThreats): game to evaluate

    Returns:
    Integer: difference between number of player 1's stacked threats and number of player 2's stacked threats.
    """
    hole_board = np.zeros((2, game.rows, game.cols))
    stacked_threat_diff = 0

    # compute map of threat hole positions
    for player in [-1, 1]:
      for threat in game.threats[player]:
        player_index = int(max(0, threat.player))
        hole_board[player_index, threat.hole_r, threat.hole_c] = 1
    
    # identify and count stacked threat holes
    for player in [-1, 1]:
      player_index = max(0, player)
      opponent_index = 1 - player_index
      for col in range(game.cols):
        for row in range(5, 0, -1):
          if (hole_board[player_index, row, col] == 1
              and hole_board[player_index, row - 1, col] == 1
              and hole_board[opponent_index, row, col] == 0):
            stacked_threat_diff += player
            break
    return stacked_threat_diff

  def map_to_asymptotic(self, num, bounds):
    """
    Applies an exponential transformation to map the given number to lie within the specified bounds.
    If num is greater than zero, the output will be at least equal to the min bound.
    As num increases, the output asymptotically approaches the max bound.
    This approach characterizes the diminishing returns of having multiple threats of the same type.

    Args:
    num (Integer): the number which is to be mapped within the specified bounds
    bounds (Tuple[Float, Float]): tuple of the form (min, max) representing the bounds within which num must be mapped

    Returns:
    Float: computed mapping of input num
    """
    if (num == 0):
      return 0
    high, low = bounds
    num -= 1
    diff = high - low
    reward = high - (diff * 0.5 ** num)
    return reward

  def is_high_reward(self, threat):
    """
    Checks whether a given Threat is a "high" threat, meaning it is potentially game-winning in a zugzwang (forced-move) situation.
    To be a high threat, it must either belong to the first player and have an empty spot on an odd row, 
    or belong to the second player and have the empty space on an even row.

    Args:
    threat (Threat): the Threat object to check

    Returns:
    Boolean: whether or not the threat is a "high" threat
    """
    return ((threat.player == 1 and threat.hole_c % 2 == 1) 
            or (threat.player == -1 and threat.hole_c % 2 == 0))