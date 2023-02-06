from src.ConnectFour import ConnectFour
from src.players.AlphaBetaPlayer import AlphaBetaPlayer
from src.players.MCTSPlayer import MCTSPlayer
from src.players.AlphaZeroPlayer import AlphaZeroPlayer
from src.players.alphazero.BufferedModelWrapper import BufferedModelWrapper
from src.players.alphazero.AlphaZeroNets import AlphaZeroCNN
from parse_utils import parse_args_single_player
from validation_utils import AgentType
import timeit
import numpy as np
from torch import zeros
import random


def play_single_player(player, human_first):
  """
  Plays a single-player game, pitting the user against the selected agent.

  Args:
  agent (Union[RandomPlayer, AlphaBetaPlayer, MCTSPlayer, AlphaZeroPlayer]): agent to play against
  human_first (Boolean): whether the human goes first
  """
  cnct4 = ConnectFour()
  is_over = False
  winner = 0
  if (not human_first):
    cnct4.perform_move(player.pick_move(cnct4))
  cnct4.print_board()
  while(not is_over):
    try: 
      col = int(input(f"Move: ")) - 1
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

def initialize_agent(agent_args):
  """
  Initializes and instance RandomPlayer, AlphaBetaPlayer, MCTSPlayer, or AlphaZeroPlayer according to provided command line arguments.

  Args:
  agent_args (Dict[String, Any]): dictionary of agent arguments

  Return:
  Union[RandomPlayer, AlphaBetaPlayer, MCTSPlayer, AlphaZeroPlayer]
  """
  agent_type = AgentType(agent_args.pop("agent"))

  if (agent_type == AgentType.RANDOM):
    return RandomPlayer(**agent_args)
  elif (agent_type == AgentType.ALPHABETA):
    return AlphaBetaPlayer(**agent_args)
  elif (agent_type == AgentType.MCTS):
    gaussian = agent_args.pop("gaussian")
    return MCTSPlayer(agent_args, gaussian)
  elif (agent_type == AgentType.ALPHAZERO):
    use_cuda = agent_args.pop("cuda")
    checkpoint_path = agent_args.pop("checkpoint")
    max_buffer_size = agent_args.pop("max_buffer_size")
    max_wait_time = agent_args.pop("max_wait_time")

    device = "cuda" if use_cuda else "cpu"
    model = AlphaZeroCNN.load_from_checkpoint(checkpoint_path).to(device)
    buffered_model = BufferedModelWrapper(model, max_buffer_size, max_wait_time)
    return AlphaZeroPlayer(buffered_model, agent_args)
  raise ValueError(f"Cannot load agent of type {agent_type}")


if __name__ == "__main__":
  args = parse_args_single_player()
  
  human_first = args.pop("human_first")
  if (human_first is None):
    human_first = bool(random.getrandbits(1))
  player = initialize_agent(args)

  play_single_player(player, human_first)

  if (isinstance(player, AlphaZeroPlayer)):
    player.close()
