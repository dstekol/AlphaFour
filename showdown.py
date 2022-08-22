from src.players.MCTSPlayer import MCTSPlayer
from src.ConnectFour import ConnectFour
from src.players.AlphaZeroPlayer import AlphaZeroPlayer
from src.players.components.AlphaZeroNets import AlphaZeroCNN
from src.players.components.EvalBuffer import EvalBuffer
import random
import pickle as pkl
from tqdm import tqdm

#random.seed(2)

trajectories = []
#filename="2000game_data.pkl"
temp = 0.5
mcts_iters = 500
#log_freq = 10
pbar = tqdm()
i = 0
win_counter = 0
first_winner_counter = 0

subopt_counter = 0
subopt_summer = 0
subopt_total = 0

mcts_args = {
            "discount": 0.96,
            "mcts_iters": 250,
            "explore_coeff": 1,
            "temperature":  0.1,
            "dirichlet_coeff": 0.02,
            "dirichlet_alpha": 0.3,
            "num_threads": 15
            }

model1 = AlphaZeroCNN(0, 0, 0).cuda()
buffer1 = EvalBuffer(model1, 5, 0.5)
model2 = AlphaZeroCNN.load_from_checkpoint("checkpoints_v5/0.ckpt").cuda()
buffer2 = EvalBuffer(model2, 5, 0.5)
num_moves = 0

num_games = 100
drop_move = 0

for i in tqdm(range(num_games)):
  #if (i > 0):
  #  print("\nProportion")
  #  print(float(win_counter) / float(i))
  #player1 = MCTSPlayer(mcts_iters=mcts_iters, temperature=temp, discount=1)
  #player2 = MCTSPlayer(mcts_iters=mcts_iters, temperature=temp, discount=0.96)
  player1 = AlphaZeroPlayer(buffer1, mcts_args)
  player2 = AlphaZeroPlayer(buffer2, mcts_args)
  cnct4 = ConnectFour()
  is_over = False
  moves = []
  scores = []
  if (bool(random.getrandbits(1))):
    #print("small going first")
    player1, player2 = player2, player1
  else:
    pass
    #print("big going first")
  while (not is_over):
    #if (cnct4.num_moves == drop_move):
    #  print("dropping")
    #  player1.drop_temperature()
    #  player2.drop_temperature()
    move = player1.pick_move(cnct4)
    moves.append(move)
    scores.append(player1.mcts.current_action_scores(cnct4))
    
    if (move != scores[-1].argmax()):
      subopt_counter += 1
      d = max(scores[-1]) - scores[-1][move]
      subopt_summer += d
      if (d >= 0.2):
        cnct4.print_board()
        print(scores[-1].round(4))
        print(f"argmax: {scores[-1].argmax()}")
        print(f"move: {move}")
    subopt_total += 1
    cnct4.perform_move(move)
    player1, player2 = player2, player1
    #cnct4.print_board()
    #print(scores[-1].round(4))
    #print(f"argmax: {scores[-1].argmax()}")
    is_over, winner = cnct4.game_over()
  num_moves += cnct4.num_moves
  if (player2.buffer == buffer2): # player 1 won
    #print("Win")
    win_counter += 1
  #else:
  #  print("Lose")
  if (winner == 1):
    first_winner_counter += 1
  trajectories.append((moves, scores, winner))
  pbar.update()
  #if ( i % log_freq == 0):
  #  pkl.dump(trajectories, open(filename, "wb"))
  #i += 1
#pkl.dump(trajectories, open(filename, "wb")
print(f"\nNum wins: {win_counter}")
print(f"\nNum first wins: {first_winner_counter}")
print(f"\nAvg num moves: {num_moves / num_games}")
print(f"drop move: {drop_move}")
print(f"subopt moves: {subopt_counter / subopt_total}")
print(f"temp: {mcts_args['temperature']}")
print(f"noise: {mcts_args['dirichlet_coeff']}")
print(f"subopt diff: {subopt_summer / subopt_counter}")
buffer1.close()
buffer2.close()