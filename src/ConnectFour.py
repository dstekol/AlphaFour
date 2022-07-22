import numpy as np
from IPython.display import clear_output
import copy
import torch

class ConnectFour:
  def __init__(self, board=None):
    self.rows = 6
    self.cols = 7
    self.last_col = None
    self.board = np.zeros((self.rows, self.cols)) if board is None else board
    self.level = np.full((self.cols), self.rows - 1) if board is None else self._compute_level()
    self.num_moves = self._compute_num_moves()
    self.player = 1 if self.num_moves % 2 == 0 else -1
    self.dirs = [(-1,  1), # top right
                 ( 0,  1), # right
                 ( 1,  1), # bottom right
                 ( 1, -1), # bottom left
                 ( 0, -1), # left
                 (-1, -1)] # top left 

  def _compute_level(self):
    return self.rows - 1 - np.count_nonzero(self.board, axis=0)

  def game_over(self):
    if (self.num_moves < 7):
      return (False, 0)
    c = self.last_col
    r = self.level[c] + 1
    if (r <= 2 and (self.board[r:r+4, c]==-self.player).all()):
      return (True, -self.player)
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
    if (   dir_count[(-1, -1)] + dir_count[(1,  1)] >=3
        or dir_count[( 0, -1)] + dir_count[(0,  1)] >=3
        or dir_count[(-1,  1)] + dir_count[(1, -1)] >=3):
      return (True, -self.player)
    if (self.num_moves == self.rows * self.cols):
      return (True, 0)
    return (False, 0)

  def perform_move(self, col, validate=False):
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
    return copy.deepcopy(self)

  def _compute_num_moves(self):
    return np.count_nonzero(self.board)

  def valid_moves(self):
    return [col for col in range(self.cols) if self.level[col] >= 0]

  def print_board(self):
    clear_output()
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

