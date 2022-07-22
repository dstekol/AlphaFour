from src.players.ConnectFourPlayer import ConnectFourPlayer
from src.players.components.AlphaZeroNets import AlphaZeroFCN

#def eval_func_template(model, game, actions):
#  action_values, state_value = model([game.board])
#  action_values = action_values.squeeze()[actions].numpy()
#  state_value = state_value.item()
#  return action_values, state_value

def eval_func(model, game, actions):
  x = torch.tensor(one_hot_state(game)).unsqueeze(0).to(model.device)
  y = model(x).cpu().squeeze().numpy()
  actions_vals = y[:-1][actions]
  state_val = y[-1]
  return action_vals, state_val

class MCTSPlayer(ConnectFourPlayer):
  def __init__(self, model_path, c_puct=1, num_iters = 1000):
    super().__init__()
    self.num_iters = num_iters
    model = AlphaZeroFCN.load_from_checkpoint(model_path)
    self.mcts = MCTS(eval_func=lambda game, actions: eval_func(model, game, actions), c_puct=c_puct)

  def pick_move(self, game):
    return self.mcts.search(game, self.num_iters)