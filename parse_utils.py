import argparse
from validation_utils import *
from pathlib import Path
from src.players.RandomPlayer import RandomPlayer
from src.players.AlphaBetaPlayer import AlphaBetaPlayer
from src.players.MCTSPlayer import MCTSPlayer
from src.players.AlphaZeroPlayer import AlphaZeroPlayer

def copy_args(args, arg_names):
  return {arg_name: args[arg_name] for arg_name in arg_names}

def parse_args_trainer():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
  parser.add_argument("--cuda", default=True, type=bool, help="Whether to use CUDA acceleration")
  parser.add_argument("--rounds", default=20, type=positive_int, help="Number of policy improvement rounds")
  parser.add_argument("--games-per-round", default=2000, type=positive_int, help="Number of self-play games to execute per policy improvement round")
  parser.add_argument("--eval-games", default=100, type=positive_int, help="Number of evaluation games to play between newly trained model and previous best model")
  parser.add_argument("--win-threshold", default=0.55, type=constrained_float, help="Fraction of evaluation games that newly trained model must win to replace previous best model")
  parser.add_argument("--base-dir", required=True, type=path_arg, help="Directory for model checkpoints (if checkpoints already exist in this directory, the trainer will use them as a starting point)")
  parser.add_argument("--flip-prob", default=0.5, type=constrained_float, help="Probability that input is flipped while training neural network (for data augmentation). If set to 0, no samples will be flipped; if set to 1, all samples will be flipped.")
  parser.add_argument("--samples-per-game", default=None, type=float, help="Number of state-outcome pairs per game trajectory to sample for training. If None, all data will be used. If in range (0,1), corresponding fraction of trajectories will be used. If integer greater than or equal to 1, corresponding number of games per trajectory will be used.")
  parser.add_argument("--validation-games", default=0.05, type=float, help="Number of self-play games to hold back for validation during neural network training. If in range (0,1), corresponding fraction of trajectories will be used. If integer greater than or equal to 1, corresponding number of games per trajectory will be used.")
  parser.add_argument("--max-queue-len", default=6000, type=positive_int, help="Maximum number of self-play games to retain in the training queue")
  parser.add_argument("--max-buffer-size", default=20, type=positive_int, help="Maximum GPU buffer size (should be at most number of threads)")
  parser.add_argument("--max-wait-time", default=0.05, type=float, help="Maximum amount of time  (in milliseconds) to wait before flushing GPU buffer")

  net_args = parser.add_argument_group("net_args")
  net_args.add_argument("--max-epochs", default=15, type=positive_int, help="Max number of backpropagation epochs to train model on data collected from each round")
  net_args.add_argument("--min-epochs", default=5, type=positive_int, help="Min number of backpropagation epochs to train model on data collected from each round")
  net_args.add_argument("--batch-size", default=100, type=positive_int, help="Batch size for training neural network")
  net_args.add_argument("--state-value-weight", default=0.5, type=constrained_float, help="Weight to put on state value prediction relative to policy prediction (1 means all weight on state value, 0 means all weight on policy)")
  net_args.add_argument("--lr", default=1e-3, type=float, help="Learning rate for training neural network")
  net_args.add_argument("--l2-reg", default=1e-5, type=float, help="Strength of L2 regularization for neural network")
  net_args.add_argument("--patience", default=None, type=int, help="Number of non-improving steps to wait before stopping training. If None, training will not be stopped even if loss is not decreasing.")
  net_args.add_argument("--train-attempts", default=4, type=positive_int, help="Number of training runs to perform at each iteration (best model is selected based on validation loss)")

  game_args = parser.add_argument_group("game_args")
  game_args.add_argument("--num-threads", default=45, type=positive_int, help="Number of threads for MCTS")
  game_args.add_argument("--mcts-iters", default=250, type=positive_int, help=" Number of MCTS/PUCT rollout simulations to execute for each move")
  game_args.add_argument("--discount", default=0.96, type=constrained_float, help="Per-step discount factor for rewards (to encourage winning quickly)")
  game_args.add_argument("--explore-coeff", default=1, type=float, help="Exploration coefficient for MCTS/PUCT search")
  game_args.add_argument("--temperature", default=0.8, type=float, help="MCTS/PUCT exploration temperature setting (before temp-drop step)")
  game_args.add_argument("--drop-temperature", default=0.05, type=float, help="MCTS/PUCT exploration temperature (after temp-drop step)")
  game_args.add_argument("--dirichlet-coeff", default=0.02, type=constrained_float, help="Dirichlet noise coefficient (added to action scores). If 0, no dirichlet noise will be added to MCTS scores; if 1, only dirichlet noise will be used.")
  game_args.add_argument("--dirichlet-alpha", default=0.3, type=float, help="Dirichlet noise distribution parameter")
  game_args.add_argument("--temp-drop-step", default=7, type=nonnegative_int, help="The episode step at which to drop the temperature (exploration strength) during self-play training games. This encourages exploration early in the game and stronger play later in the game.")
  game_args.add_argument("--resign-threshold", default=-0.85, type=float, help="Threshold value for resignation. The agent will resign if all children of the current node have values below the threshold.")
  game_args.add_argument("--resign-forbid-prob", default=0.1, type=constrained_float, help="Probability that resignation will not be allowed (used in training to prevent false positives)")

  arg_groups = [net_args, game_args]
  #return to_heirarchical_dict(parser.parse_args(), arg_groups)
  return {"seed": 42,
          "cuda": True,
          "rounds": 5,
          "games_per_round": 20,
          "eval_games": 10,
          "win_threshold": 0.55,
          "base_dir": Path("testrun1"),
          "flip_prob": 0.5,
          "samples_per_game": None,
          "validation_games": 0.10,
          "max_queue_len": 6000,
          "max_buffer_size": 20,
          "max_wait_time": 0.05,
          "net_args": {
            "patience": None,
            "train_attempts": 4,
            "max_epochs": 15,
            "min_epochs": 5,
            "batch_size": 100,
            "state_value_weight": 0.5,
            "lr": 1e-3,
            "l2_reg": 1e-5
            },
          "game_args": {
            "num_threads": 45,
            "discount": 0.96,
            "mcts_iters": 100,
            "explore_coeff": 1,
            "temperature": 0.8,
            "drop_temperature": 0.05,
            "temp_drop_step": 7,
            "resign_threshold": -0.85,
            "resign_forbid_prob": 0.1,
            "dirichlet_coeff": 0.02,
            "dirichlet_alpha": 0.3
            }
    }

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def get_single_player_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--agent", required=True, type=valid_opponent, help="Opponent agent to play against. One of: random, alphabeta, mcts, alphazero")
  parser.add_argument("--human-first", default=None, type=bool, help="Whether the human should go first. If None, first player will be chosen randomly")

  parser.add_argument("--gaussian", default=True, type=bool, help="Whether to use a Gaussian prior when choosing moves (thus biasing moves toward the center, which is generally slightly better). If set to False, a uniform distribution will be used instead.")

  parser.add_argument("--quick-search-depth", default=4, type=positive_int, help="Maximum tree depth when performing a quick-search (checking for obvious moves at the start of each turn)")
  parser.add_argument("--max-depth", default=7, type=positive_int, help="Maximum tree depth to descend to before applying heuristic evaluation functions")
  parser.add_argument("--discount", default=0.96, type=constrained_float, help="Per-step discount factor for rewards (to encourage winning quickly)")

  parser.add_argument("--num-threads", default=45, type=positive_int, help="Number of threads for MCTS")
  parser.add_argument("--mcts-iters", default=4000, type=positive_int, help=" Number of MCTS/PUCT rollout simulations to execute for each move")
  parser.add_argument("--explore-coeff", default=1, type=float, help="Exploration coefficient for MCTS/PUCT search")
  parser.add_argument("--temperature", default=0, type=float, help="MCTS/PUCT exploration temperature setting (before temp-drop step)")
  parser.add_argument("--dirichlet-coeff", default=0, type=constrained_float, help="Dirichlet noise coefficient (added to action scores). If 0, no dirichlet noise will be added to MCTS scores; if 1, only dirichlet noise will be used.")
  parser.add_argument("--dirichlet-alpha", default=0.3, type=float, help="Dirichlet noise distribution parameter")
  
  parser.add_argument("--cuda", default=True, type=bool, help="Whether to use CUDA acceleration")
  parser.add_argument("--checkpoint", type=path_arg, help="Path to saved model checkpoint")
  parser.add_argument("--max-buffer-size", default=20, type=positive_int, help="Maximum GPU buffer size (should be at most number of threads)")
  parser.add_argument("--max-wait-time", default=0.05, type=float, help="Maximum amount of time  (in milliseconds) to wait before flushing GPU buffer")

  return parser

def filter_player_args(args):
  agent_type = args["agent"]
  agent_args = {"random": ["gaussian"],
                "alphabeta": ["quick_search_depth", "max_depth", "discount"],
                "mcts": ["gaussian", "discount", "num_threads", "mcts_iters", "explore_coeff", "temperature", "dirichlet_coeff", "dirichlet_alpha"],
                "alphazero": ["max_buffer_size", "max_wait_time", "cuda", "checkpoint", "discount", "num_threads", "mcts_iters", "explore_coeff", "temperature", "dirichlet_coeff", "dirichlet_alpha"]
                }
  filtered_args = {key: args[key] for key in agent_args[agent_type]}
  filtered_args["agent"] = agent_type
  return filtered_args


def parse_args_single_player(): 
  parser = get_single_player_parser()
  args = _to_heirarchical_dict(parser.parse_args(), [])
  filtered_args = filter_player_args(args)
  filtered_args["human_first"] = args["human_first"]
  return filtered_args

def get_agent_file_args(path):
  parser = get_single_player_parser()
  with open(path, "r") as f:
    arg_text = f.read().split()
    try:
      args = parser.parse_args(arg_text)
    except:
      raise argparse.ArgumentTypeError(f"Unable to parse argument file {str(path)}. \
      Try running these arguments with run_single_player.py to determine which argument is invalid.")
  return _to_heirarchical_dict(args, [])

def parse_args_showdown():
  parser = argparse.ArgumentParser()
  parser.add_argument("--agent1args", required=True, type=path_arg, help="Path to file containing agent 1 args")
  parser.add_argument("--agent2args", required=True, type=path_arg, help="Path to file containing agent 2 args")
  
  args = parser.parse_args()
  agent_1_args = filter_player_args(get_agent_file_args(args.agent1args))
  agent_2_args = filter_player_args(get_agent_file_args(args.agent2args))
  return {"agent1args": agent_1_args, "agent2args": agent_2_args}




def _to_heirarchical_dict(args, arg_groups):
  args_dict = vars(args)
  for arg_group in arg_groups:
    title = arg_group.__dict__["title"]
    args_dict[title] = dict()
    for a in arg_group.__dict__["_group_actions"]:
      args_dict[title][a.dest] = args_dict[a.dest]
      del args_dict[a.dest]
  return args_dict