from src.ConnectFour import ConnectFour
from src.players.AlphaZeroPlayer import AlphaZeroPlayer
from run_single_player import initialize_agent
from parse_utils import parse_args_showdown

def showdown(player1, player2):
  cnct4 = ConnectFour()
  is_over = False
  playerA, playerB = player1, player2
  while (not is_over):
    move = playerA.pick_move(cnct4)
    cnct4.perform_move(move)
    player1, player2 = player2, player1
    cnct4.print_board()
    is_over, winner = cnct4.game_over()
    if (not is_over):
      input("Press Enter for next move")
  if (winner == 1):
    print("Player 1 Wins")
  elif (winner == -1) :
    print("Player 2 Wins")
  else:
    print("Tie")
  
if __name__ == "__main__":
  args = parse_args_showdown()
  player1 = initialize_agent(args["agent1args"])
  player2 = initialize_agent(args["agent2args"])
  showdown(player1, player2)
  for player in [player1, player2]:
    if (isinstance(player, AlphaZeroPlayer)):
      player.close()
