from src.players.RandomPlayer import RandomPlayer
from src.ConnectFour import ConnectFour
from src.players.alphabeta.AlphaBetaScorer import AlphaBetaScorer
from src.players.alphabeta.ConnectFourWithThreats import ConnectFourWithThreats
import math
import numpy as np
from tqdm import tqdm
import copy

class LeafScore:
  """
  Data class for storing evaluation scores for leaves in search tree.
  """
  def __init__(self, winner, score, col):
    self.winner = winner
    self.score = score
    self.col = col

  def totuple(self):
    return (self.winner, self.score, self.col)
  

class AlphaBetaPlayer:
  """
  Plays Connect Four via depth-limited tree search with alpha-beta pruning and a heuristic evaluation function.
  """
  def __init__(self,
               max_depth = 6, 
               quick_search_depth = 4,
               discount = 0.99,
               win_reward = 1, 
               low_threat_reward_bounds = (0.1, 0.2),
               high_threat_reward_bounds = (0.2, 0.3),
               stacked_threat_reward_bounds = (0.3, 0.4)):
    """
    Args:
    max_depth (Integer): maximum tree depth to descend to before applying heuristic evaluation functions
    quick_search_depth (Integer): maximum tree depth when performing a quick-search (checking for obvious moves at the start of each turn)
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
    super(AlphaBetaPlayer, self).__init__()
    self.max_depth = max_depth
    self.quick_search_depth = quick_search_depth
    self.scorer = AlphaBetaScorer(discount,
                                  win_reward,
                                  low_threat_reward_bounds, 
                                  high_threat_reward_bounds, 
                                  stacked_threat_reward_bounds)
    self.memos = {}
    self.hits = 0

  def pick_move(self, game):
    """
    Selects best move

    Args:
    game (ConnectFour)

    Returns:
    Integer: best move column
    """
    if (not isinstance(game, ConnectFourWithThreats)):
      game = ConnectFourWithThreats(game)

    # start with quick search (in case of obvious best moves)
    best_result_quick = self.quick_search(game)
    if (best_result_quick is not None):
        return best_result_quick.col

    # reset tracking vars
    self.memos = {}
    self.hits = 0

    # perform full search
    best_result = self.search(game, depth=0, max_depth=self.max_depth)
    return best_result.col
    
  def quick_search(self, game):
    """
    Performs abridged search, and returns the best move if it is obvious (ex. game-winning);
    otherwise returns None (if no move is obviously the best one)

    Args:
    game (ConnectFour)

    Returns:
    Optional[Integer]
    """
    results = []
    saved_last_col = game.last_col
    saved_threats = copy.deepcopy(game.threats)

    # recursively explore possible columns
    for col in game.valid_moves():
      game.perform_move(col)
      result = self.search(game, depth=1, max_depth=self.quick_search_depth)
      self.reverse_last_move(game, saved_last_col, saved_threats)
      result.col = col
      results += [result]
    best_result_quick = self.select_best_result(results, game)

    # return best move if game is just beginning, or if the move achieves a win or avoids a loss
    if (game._compute_num_moves() <= 2
        or self.max_depth <= self.quick_search_depth
        or best_result_quick.winner != 0
        or (len(results) > 1 and results[1].winner != 0)):
        return best_result_quick
    return None

  def reverse_last_move(self, game, saved_last_col, saved_threats):
    """
    Undoes last move, and restores previous state.
    This backtracking is a faster alternative to instantiating a new ConnectFour object for each branch of the recursive tree search

    Args:
    game (ConnectFour): current game (which needs to be backtracked)
    saved_last_col (Integer): previous value of game.last_col, before most recent move was performed 
      (in other words, the second-to-last move)
    saved_threats (Dict[Integer, List[Threat]]): previous value of game.threats
    """
    if (game.last_col is None):
      raise ValueError("No move to reverse")
    game.level[game.last_col] += 1
    game.board[game.level[game.last_col], game.last_col] = 0
    game.player *= -1
    game.last_col = saved_last_col
    game.threats = saved_threats


  def search(self, game, depth, max_depth, alpha=-math.inf, beta=math.inf):
    """
    Recursive alpha-beta tree search.

    Args:
    game (ConnectFourWithThreats): game to perform search on
    max_depth (Integer): maximum tree depth to descend to before applying heuristic evaluation functions
      (decreases as recursion depth increases)
    alpha (Float): lower bound on possible reward
    beta (Float): upper bound on possible reward

    Returns:
    List[LeafScore]: List of LeafScore objects corresponding to each current possible move
    """
    # base case when precomputed state reached
    board_hash = hash(bytes(game.board))
    if (board_hash in self.memos):
      self.hits += 1
      return LeafScore(*self.memos[board_hash])
    is_over, winner = game.game_over()
    
    # base case when game over
    if (is_over):  
      result = LeafScore(winner=winner, 
                       score=self.scorer.score(game, winner, depth), 
                       col=None)
      self.memos[board_hash] = result.totuple()
      return result
    
    # base case when max depth reaced
    if (depth == max_depth):
      result = LeafScore(winner=winner, 
                       score=self.scorer.score(game, winner, depth), 
                       col=None)
      self.memos[board_hash] = result.totuple()
      return result

    # get center-first column iterator
    valid_cols = game.valid_moves()
    self.reorder_cols(valid_cols, middle_col = game.cols // 2)
    col_iter = tqdm(valid_cols) if depth == 0 else valid_cols

    # save current game state
    saved_last_col = game.last_col
    saved_threats = copy.deepcopy(game.threats)

    # iterate recursively through columns
    results = []
    for col in col_iter:
      game_copy = game.copy()
      game.perform_move(col)
      result = self.search(game, depth + 1, max_depth, alpha, beta)
      self.reverse_last_move(game, saved_last_col, saved_threats)
      result.col = col
      results += [result]

      # discard search branch if min or max bounds on outcome exceeded
      if ((game.player == 1 and result.score > beta) 
          or (game.player == -1 and result.score < alpha)):
        break

      # update min and max bounds
      if (game.player == 1):
        alpha = max(alpha, result.score)
      else:
        beta = min(beta, result.score)

    # select best move for current player
    best_result = self.select_best_result(results, game)
    if (depth == 0):
      sorted_results = results[:]
      sorted_results.sort(key=lambda result: result.col)
    if (best_result is None):
        best_result = self.select_best_result(results, game)
    self.memos[board_hash] = best_result.totuple()
    return best_result

  def reorder_cols(self, cols, middle_col):
    """
    Reorders list of columns to encourage exploration order to start from middle

    cols (List[Integer]): list of valid (non-filled) columns
    middle_col (Integer): center-most column

    Returns:
    List[Integer]: reordered list of columns
    """
    cols.sort(key = lambda col: abs(middle_col - col))

  def select_best_result(self, results, game):
    """
    Selects the best possible move given a list of possible moves and corresponding scores

    Args:
    results (List[LeafScore]): List of LeafScore objects corresponding to each current possible move
    game (ConnectFourWithThreats): game for which to select move

    Returns:
    LeafScore: object corresponding to selected column
    """
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
    """
    Selects move randomly from list of valid moves

    Args:
    results (List[LeafScore]): List of LeafScore objects corresponding to each current possible move

    Returns:
    LeafScore: object corresponding to selected column
    """
    results = list(results)
    cols = [result.col for result in results]
    move_col = RandomPlayer.pick_random_gaussian(cols)
    for result in results:
      if (result.col == move_col):
        return result

  def select_best_move_losing(self, results, game):
    """
    Selects the best possible move in situations where the agent is theoretically guaranteed to lose.
    Avoids the problem of the agent "giving up" when all scores are equally low, by ensuring the agent still selects "best-effort" move.
    This function blocks one of the columns in which the opponent could win next turn.

    Args:
    results (List[LeafScore]): List of LeafScore objects corresponding to each current possible move
    game (ConnectFourWithThreats): game for which to select move

    Returns:
    LeafScore: object corresponding to selected column
    """
    col_counts = {}
    for result in results:
       # estimate probability that opponent will choose each column for next move
      hypo_game = game.copy()
      hypo_game.perform_move(result.col)
      opponent_move = self.search(hypo_game, depth = 1, max_depth = 2)
      if (opponent_move.col not in col_counts):
        col_counts[opponent_move.col] = 0
      col_counts[opponent_move.col] += 1

    # select column which opponent is most likely to pick next move
    max_count = max(col_counts.values())
    for result in results:
      if (result.col in col_counts and col_counts[result.col] == max_count):
        return result
    return results[0]
  

