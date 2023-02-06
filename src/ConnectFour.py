import numpy as np
import copy
import torch

class ConnectFour:
  """
  Connect Four game implementation
  """

  def __init__(self, board=None):
    self.rows = 6
    self.cols = 7
    # most recent move (None if game just started)
    self.last_col = None

    # game state (1 for player 1's pieces, -1 for player 2's pieces, 0 for unfilled positions)
    self.board = np.zeros((self.rows, self.cols), dtype=int) if board is None else board
    
    # row/height of next piece to be placed in each column
    self.level = np.full((self.cols), self.rows - 1) if board is None else self._compute_level()
    
    # number of moves performed so far
    self.num_moves = self._compute_num_moves()

    # current player
    self.player = 1 if self.num_moves % 2 == 0 else -1

    self.dirs = [(-1,  1), # top right
                 ( 0,  1), # right
                 ( 1,  1), # bottom right
                 ( 1, -1), # bottom left
                 ( 0, -1), # left
                 (-1, -1)] # top left 

  def _compute_level(self):
    """
    Computes the row/height of the next piece to be inserted in each column
    """
    return self.rows - 1 - np.count_nonzero(self.board, axis=0)

  def game_over(self):
    """

    Returns:
    game_over (Boolean): whether the game has ended
    winner (Integer): 1 if player 1 won, -1 if player 2 won, 0 if tie or game unfinished
    """

    # return if not enough moves for game to be over
    if (self.num_moves < 7):
      return (False, 0)

    c = self.last_col
    r = self.level[c] + 1

    # check vertical win
    if (r <= 2 and (self.board[r:r+4, c]==-self.player).all()):
      return (True, -self.player)

    # iterate though possible directions (horizontal, diagonal up/down) and count number of contiguous pieces
    dir_count = {dir: 0 for dir in self.dirs}
    for dir in self.dirs:
      y = r
      x = c
      for offset in range(1, 4):
        y += dir[0]
        x += dir[1]
        if (dir_count[dir] == offset - 1
            and 0 <= x < self.cols
            and 0 <= y < self.rows
            and self.board[y, x] == -self.player):
          dir_count[dir] = offset
        else:
          break
    # sum piece counts in opposite directions to determine whether connect-four achieved
    if (   dir_count[(-1, -1)] + dir_count[(1,  1)] >=3
        or dir_count[( 0, -1)] + dir_count[(0,  1)] >=3
        or dir_count[(-1,  1)] + dir_count[(1, -1)] >=3):
      return (True, -self.player)
    if (self.num_moves == self.rows * self.cols):
      return (True, 0)
    return (False, 0)

  def perform_move(self, col, validate=False):
    """
    Performs move in specified column. 
    If validate set to True, checks whether the move is valid; otherwise omits validation for sake of speed.
    Raises ValueError if invalid move detected.

    Args:
    col (Integer): column to add piece to (zero-indexed)
    validate (Boolean): whether to validate selected column
    """
    if (validate):
      if (not col in range(self.cols)):
        raise ValueError("Invalid column")
      if (self.level[col] < 0):
        raise ValueError("Column full")
    self.num_moves += 1
    self.board[self.level[col], col] = self.player
    self.level[col] -= 1
    self.player *= -1
    self.last_col = col

  def copy(self):
    """
    Returns:
    ConnectFour: deep copy of this object
    """
    return copy.deepcopy(self)

  def _compute_num_moves(self):
    """
    Returns:
    Integer: Number of moves performed so far
    """
    return np.count_nonzero(self.board)

  def valid_moves(self):
    """
    Returns:
    List[Integer]: List of valid moves (non-filled columns)
    """
    return [col for col in range(self.cols) if self.level[col] >= 0]

  def print_board(self):
    """
    Prints current game state, with 1-indexed column numbers and arrow pointing to most recent move
    """
    print("  1   2   3   4   5   6   7\n")
    print("+---"*7 + "+")
    for i, row in enumerate(self.board):
      rowstr = "|"
      for item in row:
        token = " "
        if(item==1):
          token = "X"
        elif(item==-1):
          token = "O"
        rowstr += " " + token + " |"
      print(rowstr)
      print("+---"*7 + "+")
    if (self.last_col is not None):
      col_mark_arr = [" "] * 29
      col_mark_arr[2 + self.last_col * 4] = "^"
      print("".join(col_mark_arr))

