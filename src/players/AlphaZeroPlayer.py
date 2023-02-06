from src.players.mcts.MCTS import MCTS
import torch
import torch.nn.functional as F
import numpy as np

def one_hot_state(game):
  """
  Converts ConnectFour object to one-hot tensor.
  Note that all elements of second channel (index 1) is set to the current player: out[1, :, :] = player

  Args:
  game (ConnectFour)

  Returns:
  np.ndarray: 3x6x7 array with dimensions (players, rows, columns)
  """
  perspective_board = game.player * game.board
  player_ind_board = perspective_board + 1
  one_hot_board = np.eye(3)[player_ind_board]
  transposed_board = np.transpose(one_hot_board, axes=[2, 0, 1])
  transposed_board[1] = game.player
  return transposed_board

def postprocess_tensor(t):
  """
  Postprocesses neural network output:
  moves to cpu, detaches from autodiff graph, removes extra size-1 dimensions, and converts to numpy array.

  Args:
  t (torch.tensor): neural network output
  
  Returns:
  np.ndarray
  """
  return t.cpu().detach().squeeze().numpy()

def eval_func_direct(model, game):
  """
  Evaluates state of given game directly (without buffering), returning action and state values.

  Args:
  model (torch.nn.Module): neural net to use for evaluation
  game (ConnectFour): game to evaluate

  Returns:
  actions_vals (np.ndarray): (7,) array representing policy head output (action-values of possible moves)
  state_val (np.ndarray): (1,) singleton array representing state-value head output (value of current state)
  """
  with torch.no_grad():
    x = torch.tensor(one_hot_state(game), dtype=torch.float32).unsqueeze(0).to(model.device)
    action_vals, state_vals = model(x, apply_softmax=True)
    action_vals = postprocess_tensor(action_vals)
    state_vals = postprocess_tensor(state_vals) * game.player # fixed
    return action_vals, state_vals

def eval_func_buffer(buffer, game):
  """
  Evaluates state of given game (with buffering), returning action and state values.

  Args:
  model (BufferedModelWrapper): buffer wrapper around neural net to use for evaluation
  game (ConnectFour): game to evaluate

  Returns:
  actions_vals (np.ndarray): (7,) array representing policy head output (action-values of possible moves)
  state_val (np.ndarray): (1,) singleton array representing state-value head output (value of current state)
  """
  return buffer.enqueue([one_hot_state(game)])[0]


class AlphaZeroPlayer:
  def __init__(self, buffered_model, mcts_args):
    """
    Args:
    buffered_model (BufferedModelWrapper): buffer wrapper around neural net to use for evaluation
    mcts_args (Dict[String, Any]): dictionary containing arguments for MCTS object instantiation.
      Must contain: mcts_iters, discount, explore_coeff, temperature, dirichlet_coeff, dirichlet_alpha, num_threads
    """
    super().__init__()
    self.buffered_model = buffered_model
    eval_func = lambda game: eval_func_buffer(self.buffered_model, game)
    self.mcts = MCTS(eval_func=eval_func, **mcts_args)

  def update_temperature(self, temp):
    """
    Updates MCTS temperature for selecting moves. 
    Setting temperature to 0 deterministically selects max-score move, 
    whereas increasing temperature approaches a uniform distribution over moves.
    The temperature is an important factor in controlling the exploration-exploitation tradeoff.

    Args:
    temp (Float): new temperature to use
    """
    self.mcts.update_temperature(temp)

  def _get_successors(self, game):
    """
    Returns:
    List[ConnectFour]: List of all possible successors/children of current state.
    """
    successors = []
    for action in game.valid_moves():
      game_copy = game.copy()
      game_copy.perform_move(action)
      successors.append(game_copy)
    return successors

  def should_resign(self, game, resign_threshold):
    """
    Evaluates current position to determine whether to resign.
    Resigns when the state-value of current state and all possible child states is below resign_threshold

    Args:
    game (ConnectFour): game to evaluate
    resign_threshold (Float): state-value threshold for resigning

    Returns:
    Boolean: whether to resign
    """
    with torch.no_grad():
      eval_games = self._get_successors(game)
      eval_games.append(game)
      eval_states = [one_hot_state(g) for g in eval_games]
      out = self.buffered_model.enqueue(eval_states)
      action_vals, state_vals = zip(*out)
      state_vals = np.concatenate(state_vals, axis=0)
      state_vals[:-1] *= -1
      return (state_vals < resign_threshold).all()

  def pick_move(self, game):
    """
    Finds best move according to network-augmented MCTS, as described in AlphaZero paper.

    Args:
    game (ConnectFour)

    Returns:
    best_move (Integer)
    """
    return self.mcts.search(game)

  def close(self):
    """
    Closes GPU buffer and destroys queue threads
    """
    self.buffered_model.close()