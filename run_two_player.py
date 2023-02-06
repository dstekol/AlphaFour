from src.ConnectFour import ConnectFour

if __name__ == "__main__":
  is_over = False
  cnct4 = ConnectFour()
  player = 1

  while (not is_over):
    cnct4.print_board()
    try: 
      col = int(input(f"Player {2 if player==-1 else 1} move: ")) - 1
      cnct4.perform_move(col, validate=True)
    except ValueError as e:
      print(e)
      continue
    is_over, winner = cnct4.game_over()
    player *= -1

  cnct4.print_board()
  results = {1: "Player 1 Wins", 0: "Tie", -1: "Player 2 Wins"}
  print("Game Over: " + results[winner])