from src.players.ConnectFourPlayer import ConnectFourPlayer
from src.players.components.AlphaZeroNets import AlphaZeroFCN
from src.players.components.MCTS import MCTS
import torch
import torch.nn.functional as F
import numpy as np

#def eval_func_template(model, game, actions):
#  action_values, state_value = model([game.board])
#  action_values = action_values.squeeze()[actions].numpy()
#  state_value = state_value.item()
#  return action_values, state_value

def one_hot_state(game):
  return np.eye(3)[game.board + 1]

def postprocess_tensor(t):
  return t.cpu().detach().squeeze().numpy()

def eval_func(model, game, actions):
  x = torch.tensor(one_hot_state(game), dtype=torch.float32).unsqueeze(0).to(model.device)
  action_vals, state_vals = model(x)
  return postprocess_tensor(action_vals), postprocess_tensor(state_vals)

class AlphaZeroPlayer(ConnectFourPlayer):
  def __init__(self, model, mcts_args):
    super().__init__()
    self.mcts_iters = mcts_iters
    self.model = model
    self.mcts = MCTS(eval_func=lambda game, actions: eval_func(model, game, actions), **mcts_args)

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
    eval_states = self._get_successors(game)
    eval_states.append(game)
    x = torch.tensor(np.array([one_hot_state(s) for s in eval_states]), dtype=torch.float32).to(self.model.device)
    action_vals, state_vals = self.model(x)
    action_vals = torch.tanh(action_vals)
    state_vals = postprocess_tensor(state_vals)
    return (state_vals < resign_threshold).all()

  def pick_move(self, game):
    return self.mcts.search(game)