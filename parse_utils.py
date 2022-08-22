import argparse
from validation_utils import *
from pathlib import Path

def parse_args_trainer():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
  parser.add_argument("--cuda", default=True, type=bool, help="Whether to use CUDA acceleration")
  parser.add_argument("--rounds", default=2, type=positive_int, help="Number of policy improvement rounds")
  parser.add_argument("--games-per-round", default=3, type=positive_int, help="Number of self-play games to execute per policy improvement round")
  parser.add_argument("--validation-games", default=0.05, type=float, help="Number of self-play games to hold back for validation during neural network training.")
  parser.add_argument("--eval-games", default=3, type=positive_int, help="Number of evaluation games to play between newly trained model and previous best model")
  parser.add_argument("--win-threshold", default=0.55, type=constrained_float, help="Fraction of evaluation games that newly trained model must win to replace previous best model")
  parser.add_argument("--checkpoint-dir", required=True, type=valid_checkpoint_dir, help="Directory for model checkpoints (if checkpoints already exist in this directory, the trainer will use them as a starting point)")
  parser.add_argument("--games-input-file", default=None, type=str, help="File from which game trajectories should be loaded on startup (in case training was interrupted and is being restarted). If None, no trajectories will be loaded.")
  parser.add_argument("--games-output-file", default=None, type=str, help="File where game trajectories should be saved (for checkpointing). Will be overwritten on each round. If None, trajectories will not be checkpointed.")
  parser.add_argument("--flip-prob", default=0.5, type=constrained_float, help="Probability that input is flipped while training neural network (for data augmentation)")
  parser.add_argument("--max-queue-len", default=300, type=positive_int, help="Maximum number of self-play games to retain in the training queue")
  parser.add_argument("--max-buffer-size", default=6, type=positive_int, help="Maximum GPU buffer size (should be at most number of threads)")
  parser.add_argument("--max-wait-time", default=2, type=positive_int, help="Maximum amount of time  (in milliseconds) to wait before flushing GPU buffer")
  parser.add_argument("--round-start-ind", default=0, type=positive_int, help="The index of the first round (for logging purposes). Should be zero unless restarting from a checkpoint.")

  train_args = parser.add_argument_group("train_args")
  train_args.add_argument("--max-epochs", default=1, type=positive_int, help="Max number of backpropagation epochs to train model on data collected from each round")
  train_args.add_argument("--min-epochs", default=1, type=positive_int, help="Min number of backpropagation epochs to train model on data collected from each round")
  train_args.add_argument("--batch-size", default=10, type=positive_int, help="Batch size for training neural network")
  train_args.add_argument("--value-weight", default=0.5, type=float, help="Weight to put on value prediction relative to policy prediction (1 means all weight on value, 0 means all weight on policy)")
  train_args.add_argument("--lr", default=3e-4, type=float, help="Learning rate for training neural network")
  train_args.add_argument("--l2-reg", default=1e-5, type=float, help="Strength of L2 regularization for neural network")
  train_args.add_argument("--log-dir", default=".", type=str, help="Logging directory for tensorboard logs")
  train_args.add_argument("--patience", default=7, type=int, help="Number of non-improving steps to wait before stopping training")
  train_args.add_argument("--train-attempts", default=7, type=positive_int, help="Number of training runs to perform at each iteration (best model is selected based on validation loss)")

  game_args = parser.add_argument_group("game_args")
  game_args.add_argument("--num_threads", default=8, type=positive_int, help="Number of threads for MCTS")
  game_args.add_argument("--mcts-iters", default=100, type=positive_int, help="Number of PUCT simulations to execute for each move")
  game_args.add_argument("--discount", default=0.96, type=float, help="Per-step discount factor for rewards (to encourage winning quickly)")
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
          "rounds": 30,
          "games_per_round": 1500,
          "eval_games": 20,
          "win_threshold": 0.55,
          "checkpoint_dir": Path("checkpoints_v7"),
          "games_input_file": None,
          "games_output_file": "games_v7.pkl",
          "flip_prob": 0.5,
          "samples_per_game": None,
          "validation_games": 0.10,
          "max_queue_len": 4500,
          "max_buffer_size": 20,
          "max_wait_time": 0.1,
          "round_start_ind": 0,
          "train_args": {
            "patience": 15,
            "train_attempts": 5,
            "log_dir": "alphazero_logs_v7",
            "max_epochs": 15,
            "min_epochs": 5,
            "batch_size": 100,
            "value_weight": 0.5,
            "lr": 1e-3,
            "l2_reg": 1e-3
            },
          "game_args": {
            "num_threads": 45,
            "discount": 0.96,
            "mcts_iters": 250,
            "explore_coeff": 1,
            "temperature": 1,
            "temp_drop_step": 5,
            "resign_threshold": -0.85,
            "resign_forbid_prob": 0.1,
            "dirichlet_coeff": 0,
            "dirichlet_alpha": 0.3
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