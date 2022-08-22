from src.ConnectFour import ConnectFour
from src.players.AlphaBetaPlayer import AlphaBetaPlayer
import numpy as np
import timeit
cnct4 = ConnectFour()
cnct4.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1,-1],
                        [0, 0, 1, 1, 0, 1, 1],
                        [0, 0, 1,-1, 0,-1,-1],
                        [0, 1,-1, 1, 0,-1,-1]])

#cnct4.board = np.array(
#      [[ 0,  0, -1, -1,  0,  0,  0],
#       [ 0,  0, -1,  1,  0,  0,  0],
#       [ 0,  0,  1, -1,  0,  0,  0],
#       [ 0,  0,  1,  1,  0,  1,  1],
#       [ 0,  0,  1, -1,  0, -1, -1],
#       [ 0,  1, -1,  1,  1, -1, -1]])
#cnct4.print_board()
cnct4.level = np.array( [5, 4, 2, 2, 5, 1, 1])
cnct4.last_col = 6
cnct4.player = -1
player = AlphaBetaPlayer(max_depth=9)
move = player.pick_move(cnct4)
print(move)

#[0.186, 0.146, 0.192, 0.186, 0.961, 0.146, -0.093]
#-0.093
#6

#cnct4.print_board()
#is_over = False
#winner = 0
#while(not is_over):
#  col = int(input(f"Move: ")) - 1
#  try: 
#    cnct4.perform_move(col)
#    is_over, winner = cnct4.game_over()
#    cnct4.print_board()
#    if (is_over):
#      break
#    cnct4.perform_move(player.pick_move(cnct4))
#    cnct4.print_board()
#  except ValueError as e:
#    print(e)
#    continue
#  is_over, winner = cnct4.game_over()
#print("Game Over")
