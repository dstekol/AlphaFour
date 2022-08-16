from src.players.ConnectFourPlayer import ConnectFourPlayer
from src.players.components.MCTS import MCTS
import torch
import torch.nn.functional as F
import numpy as np

def one_hot_state(game):
  perspective_board = game.player * game.board
  player_ind_board = perspective_board + 1
  one_hot_board = np.eye(3)[player_ind_board]
  transposed_board = np.transpose(one_hot_board, axes=[2, 0, 1])
  return transposed_board

def postprocess_tensor(t):
  return t.cpu().detach().squeeze().numpy()

def eval_func_direct(model, game, actions):
  with torch.no_grad():
    x = torch.tensor(one_hot_state(game), dtype=torch.float32).unsqueeze(0).to(model.device)
    action_vals, state_vals = model(x, apply_softmax=True)
    action_vals = postprocess_tensor(action_vals)
    state_vals = postprocess_tensor(state_vals) * game.player # fixed
    return action_vals, state_vals

def eval_func_buffer(buffer, game, actions):
  return buffer.enqueue([one_hot_state(game)])[0]

def eval_func_test(buffer, game, actions):
  action_vals, state_vals = eval_func_buffer(buffer, game, actions)
  return action_vals, None


class AlphaZeroPlayer(ConnectFourPlayer):
  def __init__(self, buffer, mcts_args):
    super().__init__()
    self.buffer = buffer
    eval_func = lambda game, actions: eval_func_buffer(self.buffer, game, actions)
    self.mcts = MCTS(eval_func=eval_func, **mcts_args)

  def drop_temperature(self):
    self.mcts.update_temperature(0)

  def _get_successors(self, game):
    successors = []
    for action in game.valid_moves():
      game_copy = game.copy()
      game_copy.perform_move(action)
      successors.append(game_copy)
    return successors

  def should_resign(self, game, resign_threshold):
    with torch.no_grad():
      eval_states = self._get_successors(game)
      eval_states.append(game)
      action_vals, state_vals = self.buffer.enqueue(eval_states)
      #x = torch.tensor(np.array([one_hot_state(s) for s in eval_states]), dtype=torch.float32).to(self.model.device)
      #action_vals, state_vals = self.model(x)
      #state_vals = postprocess_tensor(state_vals)
      state_vals[:-1] *= -1
      return (state_vals < resign_threshold).all()

  def pick_move(self, game):
    return self.mcts.search(game)