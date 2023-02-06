from src.ConnectFour import ConnectFour
import numpy as np

class Threat:
  """
  Class representing piece configurations in Connect Four where there are three pieces in a row,
  but the fourth cannot yet be placed. Threats are central to intelligent Connect Four play.
  """

  def __init__(self, player, hole_r, hole_c):
    """
    Args:
    player (Integer): Number representing player (1 or -1) who will win if fourth piece is placed
    hole_r (Integer): row of fourth (missing) piece
    hole_c (Integer): column of fourth (missing) piece
    """
    self.hole_r = hole_r
    self.hole_c = hole_c
    self.player = player

  @classmethod
  def get_threat_if_valid(cls, threat_arr, r1, c1, r2, c2, game, player=None):
    """
    Checks whether the specified board region contains a Threat. 
    Returns a Threat object if found, or None otherwise.
    If player is specified, will ignore Threat objects belonging to other player.

    Args:
    r1 (Integer): row coordinate of one end of the potential Threat
    c1 (Integer): column coordinate of one end of the potential Threat
    r2 (Integer): row coordinate of other end of the potential Threat
    c2 (Integer): column coordinate of other end of the potential Threat
    game (ConnectFour): ConnectFour game object
    player (Optional[Integer]): player (1 or -1) to whom the Threat must belong. 
      If None, will return a Threat for either player; otherwise, will only return a Threat if it belongs to the specified player.

    Returns:
    Optional[Threat]: Threat object corresponding to specified board region, or None if no Threat found
    """
    # determine player if not specified
    if (player is None):
      nonzero_pieces = threat_arr[threat_arr != 0]
      if (len(nonzero_pieces) == 0):
          return None
      player = nonzero_pieces[0]

    # verify 3 pieces present and 1 missing
    if (not ((threat_arr == player).sum() == 3 and (threat_arr == 0).sum() == 1)):
      return None

    # check that there is a gap below the missing piece
    hole_index = threat_arr.tolist().index(0)
    hole_r = int(np.linspace(r1, r2, 4)[hole_index])
    hole_c = int(np.linspace(c1, c2, 4)[hole_index])
    if (hole_r + 1 > game.rows or game.board[hole_r, hole_c] != 0):
      return None
    return Threat(player, hole_r, hole_c)

class ConnectFourWithThreats(ConnectFour):
  """
  Subclass of ConnectFour class, with extra logic for keeping track of Threats 
  (see Threat class for more explanation)
  """

  def __init__(self, game=None):
    super(ConnectFourWithThreats, self).__init__()
    self.threats = {-1: [], 1:[]}
    if (game is not None):
      self.board = game.board.copy()
      self.last_col = game.last_col
      self.level = game.level.copy()
      self.player = game.player
      self.find_all_threats()

  def find_all_threats(self):
    """
    Finds all existing Threats on the current board
    """

    threats = []
    for c in range(3):
      # check horizontal Threats
      for r in range(4, min(self.level), -1):
        threat_arr = self.board[r, c:c+4]
        threat = Threat.get_threat_if_valid(threat_arr, r, c, r, c+3, self)
        threats += [threat] if threat is not None else []

      # check diagonal Threats
      for r in range(5, min(self.level) + 4, -1):
        # check diagonal-down
        sub_board = self.board[r-3:r+1, c:c+4]
        threat_arr = np.diag(sub_board)
        threat = Threat.get_threat_if_valid(threat_arr, r-3, c, r, c+3, self)
        threats += [threat] if threat is not None else []

        # check diagonal-up
        threat_arr = np.diag(sub_board[::-1,:])
        threat = Threat.get_threat_if_valid(threat_arr, r, c, r-3, c+3, self)
        threats += [threat] if threat is not None else []
    hole_board = np.zeros((2, self.rows, self.cols))
    for threat in threats:
      player_index = int(max(0, threat.player))
      if (hole_board[player_index, threat.hole_r, threat.hole_c] == 0):
        self.threats[threat.player].append(threat)
        hole_board[player_index, threat.hole_r, threat.hole_c] = 1

  def perform_move(self, col):
    """
    Updates state according to player move, including Threat bookkeeping.
    
    Args:
    col (Integer): column to place next piece
    """
    ConnectFour.perform_move(self, col)
    self._remove_obsolete_threats()
    self._get_new_threats()

  def _remove_obsolete_threats(self):
    """
    Removes any Threats which were previously present but have since been destroyed.
    """
    threat_filter_func = lambda threat: self.board[threat.hole_r, threat.hole_c] == 0
    for player in [-1, 1]:
      self.threats[player] = list(filter(threat_filter_func, self.threats[player]))

  def _get_new_threats(self):
    """
    Finds newly created Threats and adds them to self.threats
    """
    if (self.last_col is None):
      return
    new_threats = []
    c = self.last_col
    r = self.level[c] + 1
    for i in range(4):
      c_min = c - 3 + i
      c_max = c + i
      # check for horizontal Threats
      if (c_min in range(self.cols) and c_max in range(self.cols)):
        threat_arr = self.board[r, c_min:c_max+1]
        threat = Threat.get_threat_if_valid(threat_arr, r, c_min, r, c_max, self, -self.player)
        new_threats += [threat] if threat is not None else []

        r_min = r - 3 + i
        r_max = r + i
        # check for diagonal-down Threats
        if (r_min in range(self.rows) and r_max in range(self.rows)):
          sub_board = self.board[r_min:r_max+1, c_min:c_max+1]
          threat_arr = np.diag(sub_board)
          threat = Threat.get_threat_if_valid(threat_arr, r_min, c_min, r_max, c_max, self, -self.player)
          new_threats += [threat] if threat is not None else []
        r_min = r - i
        r_max = r + 3 - i
        # check for diagonal-up Threats
        if (r_min in range(self.rows) and r_max in range(self.rows)):
          sub_board = self.board[r_min:r_max+1, c_min:c_max+1][::-1,:]
          threat_arr = np.diag(sub_board)
          threat = Threat.get_threat_if_valid(threat_arr, r_max, c_min, r_min, c_max, self, -self.player)
          new_threats += [threat] if threat is not None else []
    
    # add only threats which haven't already been found
    hole_board = np.zeros((self.rows, self.cols))
    for threat in self.threats[-self.player]:
      hole_board[threat.hole_r, threat.hole_c] = 1
    for threat in new_threats:
      if (hole_board[threat.hole_r, threat.hole_c] == 0):
        self.threats[-self.player].append(threat)
        hole_board[threat.hole_r, threat.hole_c] = 1