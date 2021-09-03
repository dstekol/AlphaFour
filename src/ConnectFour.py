import numpy as np
from IPython.display import clear_output
import copy

class ConnectFour:
  def __init__(self):
    self.player = 1
    self.rows = 6
    self.cols = 7
    self.last_col = None
    self.board = np.zeros((self.rows, self.cols))
    self.level = np.full((self.cols), self.rows - 1)

  def game_over(self):
    if (self.last_col is None):
      print("last col is none")
    c = self.last_col
    r = self.level[c] + 1
    if (r <= 2 and (self.board[r:r+4, c]==-self.player).all()):
      return (True, -self.player)
    dirs = {(-1, -1): 0,
            ( 0, -1): 0,
            (-1,  1): 0,
            ( 1, -1): 0,
            ( 0,  1): 0,
            ( 1,  1): 0
           }
    for offset in range(1, 4):
      for dir in dirs:
        y = r + offset * dir[0]
        x = c + offset * dir[1]
        if (dirs[dir] == offset - 1
            and 0 <= x < self.cols
            and 0 <= y < self.rows
            and self.board[y, x] == -self.player):
          dirs[dir] = offset
    dir_pairs = [((-1, -1), (1,  1)),
                 (( 0, -1), (0,  1)),
                 ((-1,  1), (1, -1)),
                ]
    for dir_pair in dir_pairs:
      if (dirs[dir_pair[0]] + dirs[dir_pair[1]] + 1 >= 4):
        return (True, -self.player)
    if ((self.level==-1).all()):
      return (True, 0)
    return (False, 0)

  def perform_move(self, col):
    if (not col in range(self.cols)):
      raise ValueError("Invalid column")
    if (self.level[col] < 0):
      raise ValueError("Column full")
    self.board[self.level[col], col] = self.player
    self.level[col] -= 1
    self.player *= -1
    self.last_col = col

  def copy(self):
    return copy.deepcopy(self)

  def num_moves(self):
    return np.count_nonzero(self.board)

  def valid_cols(self):
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

