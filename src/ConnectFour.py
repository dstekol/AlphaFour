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
    self.level = np.full((self.cols), self.rows - 1) if board is None else self.get_level()
    self.num_moves = self._compute_num_moves()
    self.player = 1 if self.num_moves % 2 == 0 else -1

    
    self.dirs = [(-1,  1), ( 0,  1),  (1,  1), ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1)]
    #self.neighbors = np.empty(shape=(self.rows, self.cols), dtype=object)
    #for i in range(self.rows):
    #  for j in range(self.cols):
    #    self.neighbors[i, j] = {dir: 0 for dir in self.dirs}
    
    #self.dir_inds = {"top-right": 0,
    #             "right":         1,
    #             "bottom-right":  2,
    #             "bottom":        3,
    #             "bottom left":   4,
    #             "left":          5,
    #             "top left":      6}
    #self.dir_pairs = [((-1, -1), (1,  1)), 
    #                  (( 0, -1), (0,  1)),
    #                  ((-1,  1), (1, -1))]
    #self.neighbors = np.zeros((self.rows, self.cols, len(self.dirs)))

    #win_filter = torch.zeros(4, 1, 4, 4, dtype=torch.int8)
    #win_filter[0, :, :, 0] = 1 
    #win_filter[1, :, 0, :] = 1
    #win_filter[2, :, :, :] = torch.eye(4) 
    #win_filter[3, :, :, :] = torch.flip(torch.eye(4), dims=(0,)) 
    #self.win_filter = win_filter
    

  def get_level(self):
    return self.rows - 1 - np.count_nonzero(self.board, axis=0)

  def game_over(self):
    if (self.num_moves < 7):
      return (False, 0)
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
    for dir in dirs:
      y = r
      x = c
      for offset in [1,2,3]:
        y += dir[0]
        x += dir[1]
        if (dirs[dir] == offset - 1
            and 0 <= x < self.cols
            and 0 <= y < self.rows
            and self.board[y, x] == -self.player):
          dirs[dir] = offset
        else:
          break
    if (   dirs[(-1, -1)] + dirs[(1,  1)] >=3
        or dirs[( 0, -1)] + dirs[(0,  1)] >=3
        or dirs[(-1,  1)] + dirs[(1, -1)] >=3):
      return (True, -self.player)
    
    #for dir_pair in self.dir_pairs:
    #  if (dirs[dir_pair[0]] + dirs[dir_pair[1]] >= 3):
    #    return (True, -self.player)
    #if ((self.level==-1).all()):
    if (self.num_moves == 42):
      return (True, 0)
    return (False, 0)

  #def game_over(self):
  #  if (self.num_moves < 7):
  #    return (False, 0)
  #  c = self.last_col
  #  r = self.level[c] + 1
  #  cell_neighbors = self.neighbors[r, c]
  #  if (cell_neighbors[(1, 0)] >= 3):
  #    return (True, -self.player)
  #  for dir1, dir2 in self.dir_pairs:
  #    if (cell_neighbors[dir1] + cell_neighbors[dir2] >= 3):
  #      return (True, -self.player)
  #  #if ( cell_neighbors[self.dir_inds["top-left"]] + cell_neighbors[self.dir_inds["bottom-right"]] >= 3
  #  #  or cell_neighbors[self.dir_inds["top-right"]] + cell_neighbors[self.dir_inds["bottom-left"]] >= 3
  #  #  or cell_neighbors[self.dir_inds["left"]] + cell_neighbors[self.dir_inds["right"]] >= 3
  #  #  or cell_neighbors[self.dir_inds["bottom"]] >= 3):
  #  #    return (True, -self.player)
  #  if (self.num_moves == 42):
  #    return (True, 0)
  #  return (False, 0)

  def perform_move(self, col):
    # TODO move
    #if (not col in range(self.cols)):
    #  raise ValueError("Invalid column")
    #if (self.level[col] < 0):
    #  raise ValueError("Column full")
    self.num_moves += 1
    self.board[self.level[col], col] = self.player
    #self.update_neighbors(self.level[col], col)
    self.level[col] -= 1
    self.player *= -1
    self.last_col = col
    

  #def update_neighbors(self, r, c):
  #  cell_neighbors = self.neighbors[r, c]
  #  for dir in self.dirs:
  #    r_offset = r + dir[0]
  #    c_offset = c + dir[1]
  #    if (0 <= r_offset < self.rows and 0 <= c_offset < self.cols
  #        and self.board[r,c] == self.board[r_offset, c_offset]):
  #      cell_neighbors[dir] = self.neighbors[r_offset, c_offset][dir] + 1
  #      #dir_ind = self.dir_inds[dir]
  #      #cell_neighbors[dir_ind] = \
  #      #  self.neighbors[r_offset, c_offset, dir_ind] + 1
  #  for dir1, dir2 in self.dir_pairs:
  #    r_offset1 = r + dir1[0]
  #    c_offset1 = c + dir1[1]
  #    r_offset2 = r + dir2[0]
  #    c_offset2 = c + dir2[1]

  #    #dir1_ind = self.dir_inds[dir1]
  #    #dir2_ind = self.dir_inds[dir2]
  #    if (0 <= r_offset1 < self.rows and 0 <= c_offset1 < self.cols
  #        and self.board[r,c] == self.board[r_offset1, c_offset1]):
  #      #self.neighbors[r_offset1, c_offset1, dir1_ind] = \
  #      #  cell_neighbors[dir2_ind]
  #      self.neighbors[r_offset1, c_offset1][dir2] = cell_neighbors[dir1] + 1
  #    if (0 <= r_offset2 < self.rows and 0 <= c_offset2 < self.cols
  #        and self.board[r,c] == self.board[r_offset2, c_offset2]):
  #      #self.neighbors[r_offset1, c_offset1, dir2_ind] = \
  #      #  cell_neighbors[dir1_ind]
  #      self.neighbors[r_offset2, c_offset2][dir1] = cell_neighbors[dir2] + 1

  def copy(self):
    return copy.deepcopy(self)

  def _compute_num_moves(self):
    return np.count_nonzero(self.board)

  def valid_moves(self):
    return [col for col in range(self.cols) if self.level[col] >= 0]

  def board_hash(self):
    pass #TODO

  def board_one_hot(self):
    pass # TODO

  #def game_over_conv(self):
  #  if (self.num_moves < 7):
  #    return (False, 0)
  #  last_row = self.level[self.last_col]
  #  min_row = max(0, last_row - 4)
  #  max_row = min(self.rows - 1, last_row + 4) + 1
  #  min_col = max(0, self.last_col - 4)
  #  max_col = min(self.cols - 1, self.last_col + 4) + 1
  #  board = torch.tensor(self.board, dtype=torch.int8)[min_row:max_row, min_col:max_col]
  #  out = torch.nn.functional.conv2d(board.unsqueeze(0).unsqueeze(0), self.win_filter, padding=3)
  #  if (out.max() == 4):
  #    return (True, 1)
  #  elif (out.min() == -4):
  #    return (True, -1)
  #  elif (self.num_moves == self.rows * self.cols):
  #    return (True, 0)
  #  else:
  #    return (False, 0)


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

