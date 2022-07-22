from src.ConnectFour import ConnectFour
from src.players.AlphaBetaPlayer import AlphaBetaPlayer
from src.players.MCTSPlayer import MCTSPlayer
import timeit

cnct4 = ConnectFour()
#player = AlphaBetaPlayer(max_depth=8)
player = MCTSPlayer(mcts_iters=6000)
is_over = False
winner = 0
human_first = True
if (not human_first):
  #cnct4.perform_move(player.pick_move(cnct4))
  #cnct4.perform_move(3)
  t = timeit.timeit(lambda: cnct4.perform_move(player.pick_move(cnct4)), number=1)
  print("time: " + str(t))
cnct4.print_board()
while(not is_over):
  #print(len(player.mcts.nodes))
  col = int(input(f"Move: ")) - 1
  try: 
    cnct4.perform_move(col, validate=True)
    is_over, winner = cnct4.game_over()
    cnct4.print_board()
    if (is_over):
      break
    cnct4.perform_move(player.pick_move(cnct4))
    cnct4.print_board()
  except ValueError as e:
    print(e)
    continue
  is_over, winner = cnct4.game_over()
results = {1: "Player 1 Wins", 0: "Tie", -1: "Player 2 Wins"}
print("Game Over: " + results[winner])

#t = timeit.timeit(lambda: player.pick_move(cnct4), number = 1)
#print(t)
