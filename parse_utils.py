import argparse
from validation_utils import *
from pathlib import Path

def parse_args_trainer():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
  parser.add_argument("--cuda", default=True, type=bool, help="Whether to use CUDA acceleration")
  parser.add_argument("--rounds", default=2, type=positive_int, help="Number of policy improvement rounds")
  parser.add_argument("--games-per-round", default=3, type=positive_int, help="Number of self-play games to execute per policy improvement round")
  parser.add_argument("--validation-games", default=5, type=float, help="Number of self-play games to hold back for validation during neural network training. Can be integer (number of games), float between 0 and 1 (proportion of games played during round), or None (no validation games)")
  parser.add_argument("--eval-games", default=3, type=positive_int, help="Number of evaluation games to play between newly trained model and previous best model")
  parser.add_argument("--win-threshold", default=0.55, type=constrained_float, help="Fraction of evaluation games that newly trained model must win to replace previous best model")
  parser.add_argument("--checkpoint-dir", required=True, type=valid_checkpoint_dir, help="Directory for model checkpoints (if checkpoints already exist in this directory, the trainer will use them as a starting point)")
  parser.add_argument("--flip-prob", default=0.5, type=constrained_float, help="Probability that input is flipped while training neural network (for data augmentation)")
  parser.add_argument("--samples-per-game", default=None, type=float, help="Number of samples from each self-play game to train neural network on. Can be integer (number of samples), float between 0 and 1 (proportion of game state samples), or None (all samples).")

  train_args = parser.add_argument_group("train_args")
  train_args.add_argument("--epochs-per-round", default=1, type=positive_int, help="How many epochs to train model on data collected from each round")
  train_args.add_argument("--batch-size", default=10, type=positive_int, help="Batch size for training neural network")
  train_args.add_argument("--policy-weight", default=1, type=positive_int, help="Batch size for training neural network")
  train_args.add_argument("--value-weight", default=1, type=positive_int, help="Batch size for training neural network")
  train_args.add_argument("--lr", default=3e-4, type=float, help="Learning rate for training neural network")
  train_args.add_argument("--l2-reg", default=1e-5, type=float, help="Strength of L2 regularization for neural network")


  game_args = parser.add_argument_group("game_args")
  game_args.add_argument("--mcts-iters", default=100, type=positive_int, help="Number of PUCT simulations to execute for each move")
  game_args.add_argument("--explore_coeff", default=1, type=float, help="Exploration coefficient for MCTS/PUCT search")
  game_args.add_argument("--temperature", default=1, type=float, help="MCTS/PUCT exploration temperature")
  game_args.add_argument("--dirichlet-coeff", default=0.25, type=float, help="Dirichlet noise coefficient (added to action scores)")
  game_args.add_argument("--dirichlet-alpha", default=0.03, type=float, help="Dirichlet noise distribution parameter")
  game_args.add_argument("--temp-drop-step", default=15, type=nonnegative_int, help="The episode step at which to drop the temperature (exploration strength) to 0 during self-play training games")
  game_args.add_argument("--resign-threshold", default=-0.85, type=float, help="Threshold value below which agent will resign")
  game_args.add_argument("--resign-forbid-prob", default=0.1, type=constrained_float, help="Probability that resignation will not be allowed (used in training to prevent false positives)")

  arg_groups = [train_args, game_args]
  #return to_heirarchical_dict(parser.parse_args(), arg_groups)
  return {"seed": 42,
          "cuda": True,
          "rounds": 20,
          "games_per_round": 100,
          "eval_games": 20,
          "win_threshold": 0.55,
          "checkpoint_dir": Path("checkpoints"),
          "flip_prob": 0.5,
          "samples_per_game": None,
          "train_args": {
            "epochs_per_round": 20,
            "batch_size": 20,
            "policy_weight": 1,
            "value_weight": 1,
            "lr": 1e-3,
            "l2_reg": 1e-5
            },
          "game_args": {
            "mcts_iters": 1000,
            "explore_coeff": 1,
            "temperature": 1,
            "temp_drop_step": 15,
            "resign_threshold": -0.85,
            "resign_forbid_prob": 0.1
            }
    }

def parse_args_single_player():
  parser = argparse.ArgumentParser()
  parser.add_argument("--opponent", type=valid_opponent, help="Opponent to play against. One of: Random, AlphaBeta, PUCT, AlphaZero")
  parser.add_argument("--mcts-iters", type=positive_int, help="Number of PUCT simulations to execute for each move (only valid for PUCT or AlphaZero opponents).")
  
  alpha_beta_args = parser.add_argument_group("alpha_beta_args")
  alpha_beta_args.add_argument("--depth", type=positive_int, help="Max depth of alpha-beta search (only valid for AlphaBeta opponent).")

  return parser.parse_args()

def _to_heirarchical_dict(args, arg_groups):
  args_dict = vars(args)
  for arg_group in arg_groups:
    title = arg_group.__dict__["title"]
    args_dict[title] = dict()
    for a in arg_group.__dict__["_group_actions"]:
      args_dict[title][a.dest] = args_dict[a.dest]
      del args_dict[a.dest]
  return args_dict