from src.players.ConnectFourPlayer import ConnectFourPlayer
from src.players.RandomPlayer import RandomPlayer
from src.ConnectFour import ConnectFour
import math
import numpy as np
from tqdm import tqdm
import copy

class LeafScore:
  def __init__(self, winner, score, col):
    self.winner = winner
    self.score = score
    self.col = col

  def totuple(self):
    return (self.winner, self.score, self.col)
  

class AlphaBetaPlayer(ConnectFourPlayer):
  def __init__(self,
               max_depth = 6, 
               quick_search_depth = 4,
               discount_factor = 0.99,
               win_reward = 1, 
               low_threat_reward_bounds = (0.1, 0.2),
               high_threat_reward_bounds = (0.2, 0.3),
               stacked_threat_reward_bounds = (0.3, 0.4)):
    super(AlphaBetaPlayer, self).__init__()
    self.max_depth = max_depth
    self.quick_search_depth = quick_search_depth
    self.scorer = AlphaBetaScorer(discount_factor,
                                  win_reward,
                                  low_threat_reward_bounds, 
                                  high_threat_reward_bounds, 
                                  stacked_threat_reward_bounds)
    self.memos = {}
    self.hits = 0

  def pick_move(self, game):
    if (not isinstance(game, ConnectFourWithThreats)):
      game = ConnectFourWithThreats(game)
    best_result_quick = self.quick_search(game)
    if (best_result_quick is not None):
        return best_result_quick.col
    self.memos = {}
    self.hits = 0
    best_result = self.search(game, depth=0, max_depth=self.max_depth)
    print(round(best_result.score, 3))
    print(len(self.memos))
    print(self.hits)
    return best_result.col
    
  def quick_search(self, game):
    results = []
    saved_last_col = game.last_col
    saved_threats = copy.deepcopy(game.threats)
    for col in game.valid_moves():
      game.perform_move(col)
      result = self.search(game, depth=1, max_depth=self.quick_search_depth)
      self.reverse_last_move(game, saved_last_col, saved_threats)
      result.col = col
      results += [result]
    best_result_quick = self.select_best_result(results, game)
    if (game.compute_num_moves() <= 2
        or self.max_depth <= self.quick_search_depth
        or best_result_quick.winner != 0
        or (len(results) > 1 and results[1].winner != 0)):
        return best_result_quick
    return None

  def reverse_last_move(self, game, saved_last_col, saved_threats):
    if (game.last_col is None):
      raise ValueError("No move to reverse")
    game.level[game.last_col] += 1
    game.board[game.level[game.last_col], game.last_col] = 0
    game.player *= -1
    game.last_col = saved_last_col
    game.threats = saved_threats


  def search(self, game, depth, max_depth, alpha=-math.inf, beta=math.inf):
    # base case
    board_hash = hash(bytes(game.board))
    if (board_hash in self.memos):
      self.hits += 1
      return LeafScore(*self.memos[board_hash])
    is_over, winner = game.game_over()
    if (is_over):  # check if game over
      result = LeafScore(winner=winner, 
                       score=self.scorer.score(game, winner, depth), 
                       col=None)
      self.memos[board_hash] = result.totuple()
      return result
    
    if (depth == max_depth):  # check if max depth reached
      result = LeafScore(winner=winner, 
                       score=self.scorer.score(game, winner, depth), 
                       col=None)
      self.memos[board_hash] = result.totuple()
      return result
    valid_cols = game.valid_moves()
    self.reorder_cols(valid_cols, middle_col = game.cols // 2)
    results = []
    col_iter = tqdm(valid_cols) if depth == 0 else valid_cols
    saved_last_col = game.last_col
    saved_threats = copy.deepcopy(game.threats)
    for col in col_iter:
      game_copy = game.copy()
      game.perform_move(col)
      result = self.search(game, depth + 1, max_depth, alpha, beta)
      self.reverse_last_move(game, saved_last_col, saved_threats)
      result.col = col
      results += [result]
      if ((game.player == 1 and result.score > beta) 
          or (game.player == -1 and result.score < alpha)):
        break
      if (game.player == 1):
        alpha = max(alpha, result.score)
      else:
        beta = min(beta, result.score)
    best_result = self.select_best_result(results, game)
    if (depth == 0):
      sorted_results = results[:]
      sorted_results.sort(key=lambda result: result.col)
      print([round(result.score, 3) for result in sorted_results])
    if (best_result is None): #debug
        best_result = self.select_best_result(results, game)
    self.memos[board_hash] = best_result.totuple()
    return best_result

  def reorder_cols(self, cols, middle_col):
    cols.sort(key = lambda col: abs(middle_col - col))

  def select_best_result(self, results, game):
    result_scorer_func = lambda result: result.score
    results.sort(key = result_scorer_func, reverse = (game.player == 1))
    top_result = results[0]
    if (top_result.winner == game.player or top_result.score == 0):
      filter_top_results = lambda result: result.score == top_result.score  
      filtered_results = list(filter(filter_top_results, results))
      return self.select_random_move(filtered_results)
    elif (top_result.winner == -game.player):
      return self.select_best_move_losing(results, game)
    else:
      return top_result

  def select_random_move(self, results):
    results = list(results)
    cols = [result.col for result in results]
    move_col = RandomPlayer.pick_random_gaussian(cols)
    for result in results:
      if (result.col == move_col):
        return result

  def select_best_move_losing(self, results, game):
    col_counts = {}
    for result in results:
      hypo_game = game.copy()
      hypo_game.perform_move(result.col)
      opponent_move = self.search(hypo_game, depth = 1, max_depth = 2)
      if (opponent_move.col not in col_counts):
        col_counts[opponent_move.col] = 0
      col_counts[opponent_move.col] += 1
    max_count = max(col_counts.values())
    for result in results:
      if (result.col in col_counts and col_counts[result.col] == max_count):
        return result
    return results[0]

class AlphaBetaScorer:
  def __init__(self,
               discount_factor,
               win_reward, 
               low_threat_reward_bounds, 
               high_threat_reward_bounds, 
               stacked_threat_reward_bounds):
    self.discount_factor = discount_factor
    self.win_reward = win_reward
    self.low_threat_reward_bounds = low_threat_reward_bounds
    self.high_threat_reward_bounds = high_threat_reward_bounds
    self.stacked_threat_reward_bounds = stacked_threat_reward_bounds

  def score(self, game, winner, depth):
    if (abs(self.base_score(game, winner)  * (self.discount_factor ** depth)) > 1): #debug
      i = 1
    return self.base_score(game, winner)  * (self.discount_factor ** depth)

  def base_score(self, game, winner):
    if (winner != 0):
      return winner * self.win_reward
    
    reward = 0
    for player in [-1, 1]:
      num_low_threats = 0
      num_high_threats = 0
      for threat in game.threats[player]:
        if (self.is_high_reward(threat)):
          num_high_threats += 1
        else:
          num_low_threats += 1
      if (num_low_threats == 5):
          i = 1
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
    hole_board = np.zeros((2, game.rows, game.cols))
    stacked_threat_diff = 0
    for player in [-1, 1]:
      for threat in game.threats[player]:
        player_index = int(max(0, threat.player))
        hole_board[player_index, threat.hole_r, threat.hole_c] = 1
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
    if (num == 0):
      return 0
    high, low = bounds
    num -= 1
    diff = high - low
    reward = high - (diff * 0.5 ** num)
    return reward

  def is_high_reward(self, threat):
    return ((threat.player == 1 and threat.hole_c % 2 == 1) 
            or (threat.player == -1 and threat.hole_c % 2 == 0))
    
class ConnectFourWithThreats(ConnectFour):
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
    threats = []
    for c in range(3):
      for r in range(4, min(self.level), -1):
        threat_arr = self.board[r, c:c+4]
        threat = Threat.get_threat_if_valid(threat_arr, r, c, r, c+3, self)
        threats += [threat] if threat is not None else []
      for r in range(5, min(self.level) + 4, -1):
        sub_board = self.board[r-3:r+1, c:c+4]
        threat_arr = np.diag(sub_board)
        threat = Threat.get_threat_if_valid(threat_arr, r-3, c, r, c+3, self)
        threats += [threat] if threat is not None else []

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
    ConnectFour.perform_move(self, col)
    self.update_threats()

  def update_threats(self):
    self.remove_obsolete_threats()
    self.get_new_threats()

  def remove_obsolete_threats(self):
    threat_filter_func = lambda threat: self.board[threat.hole_r, threat.hole_c] == 0
    for player in [-1, 1]:
      self.threats[player] = list(filter(threat_filter_func, self.threats[player]))

  def get_new_threats(self):
    if (self.last_col is None):
      return
    new_threats = []
    c = self.last_col
    r = self.level[c] + 1
    for i in range(4):
      c_min = c - 3 + i
      c_max = c + i
      if (c_min in range(self.cols) and c_max in range(self.cols)):
        threat_arr = self.board[r, c_min:c_max+1]
        threat = Threat.get_threat_if_valid(threat_arr, r, c_min, r, c_max, self, -self.player)
        new_threats += [threat] if threat is not None else []

        r_min = r - 3 + i
        r_max = r + i
        if (r_min in range(self.rows) and r_max in range(self.rows)):
          sub_board = self.board[r_min:r_max+1, c_min:c_max+1]
          threat_arr = np.diag(sub_board)
          threat = Threat.get_threat_if_valid(threat_arr, r_min, c_min, r_max, c_max, self, -self.player)
          new_threats += [threat] if threat is not None else []
        r_min = r - i
        r_max = r + 3 - i
        if (r_min in range(self.rows) and r_max in range(self.rows)):
          sub_board = self.board[r_min:r_max+1, c_min:c_max+1][::-1,:]
          threat_arr = np.diag(sub_board)
          threat = Threat.get_threat_if_valid(threat_arr, r_max, c_min, r_min, c_max, self, -self.player)
          new_threats += [threat] if threat is not None else []

    hole_board = np.zeros((self.rows, self.cols))
    for threat in self.threats[-self.player]:
      hole_board[threat.hole_r, threat.hole_c] = 1
    for threat in new_threats:
      if (hole_board[threat.hole_r, threat.hole_c] == 0):
        self.threats[-self.player].append(threat)
        hole_board[threat.hole_r, threat.hole_c] = 1

class Threat:
  def __init__(self, player, hole_r, hole_c):
    self.hole_r = hole_r
    self.hole_c = hole_c
    self.player = player

  @classmethod
  def get_threat_if_valid(cls, threat_arr, r1, c1, r2, c2, game, player=None):
    if (player is None):
      nonzero_pieces = threat_arr[threat_arr != 0]
      if (len(nonzero_pieces) == 0):
          return None
      player = nonzero_pieces[0]
    if (not ((threat_arr == player).sum() == 3 and (threat_arr == 0).sum() == 1)):
      return None
    hole_index = threat_arr.tolist().index(0)
    hole_r = int(np.linspace(r1, r2, 4)[hole_index])
    hole_c = int(np.linspace(c1, c2, 4)[hole_index])
    if (hole_r + 1 > game.rows or game.board[hole_r, hole_c] != 0):
      return None
    return Threat(player, hole_r, hole_c)
  

