from src.ConnectFour import ConnectFour
from src.players.AlphaBetaPlayer import AlphaBetaPlayer
from src.players.MCTSPlayer import MCTSPlayer
from src.players.AlphaZeroPlayer import AlphaZeroPlayer #, eval_func
from src.players.components.EvalBuffer import EvalBuffer
from src.players.components.AlphaZeroNets import AlphaZeroFCN, AlphaZeroCNN
import timeit
import numpy as np
from torch import zeros

np.set_printoptions(precision=3)

cnct4 = ConnectFour()
#player = AlphaBetaPlayer(max_depth=8)

mcts_args = {
            "discount": 0.96,
            "mcts_iters": 4000,
            "explore_coeff": 1,
            "temperature": 0,
            "dirichlet_coeff": 0,
            "dirichlet_alpha": 0.03,
            "num_threads": 30
            }

class FakeModel:
  def __init__(self):
    self.device = "cpu"

  def __call__(self, inp, apply_softmax=False):
    a = zeros(inp.shape[0], 7, device=self.device)
    a[:] = 1 / 7
    return a, zeros(inp.shape[0], 1, device=self.device)

model = FakeModel()

#player = MCTSPlayer(mcts_args)
#model = AlphaZeroFCN.load_from_checkpoint("sl_model.ckpt").cuda()
model = AlphaZeroCNN(0, 0, 0)
buffer = EvalBuffer(model, 15, 2)
player = AlphaZeroPlayer(buffer, mcts_args)
is_over = False
winner = 0
human_first = True
def print_eval(game):
  actions = game.valid_moves()
  #action_vals, state_vals = eval_func(model, game, actions)
  #print(action_vals)
  #print(state_vals)
if (not human_first):
  #cnct4.perform_move(player.pick_move(cnct4))
  #cnct4.perform_move(3)
  t = timeit.timeit(lambda: cnct4.perform_move(player.pick_move(cnct4)), number=1)
  print("time: " + str(t))
cnct4.print_board()
while(not is_over):
  print(len(player.mcts.nodes))
  #player.mcts.print_entropy()
  try: 
    print_eval(cnct4)
    col = int(input(f"Move: ")) - 1
    cnct4.perform_move(col, validate=True)
    is_over, winner = cnct4.game_over()
    cnct4.print_board()
    if (is_over):
      break
    #print_eval(cnct4)
    #move = player.pick_move(cnct4)
    #print(player.mcts.current_action_scores(cnct4, temperature=1))
    #cnct4.perform_move(move)

    t = timeit.timeit(lambda: cnct4.perform_move(player.pick_move(cnct4)), number=1)
    print("time: " + str(t))
    cnct4.print_board()
  except ValueError as e:
    print(e)
    continue
  is_over, winner = cnct4.game_over()
results = {1: "Player 1 Wins", 0: "Tie", -1: "Player 2 Wins"}
print("Game Over: " + results[winner])

buffer.close()

#t = timeit.timeit(lambda: player.pick_move(cnct4), number = 1)
#print(t)
