import argparse
from validation_utils import *

def parse_args_trainer():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
  parser.add_argument("--rounds", type=positive_int, default=10, help="Number of policy improvement rounds")
  parser.add_argument("--games-per-round", default=100, type=positive_int, help="Number of self-play games to execute per policy improvement round")
  parser.add_argument("--epochs-per-round", default=1, type=positive_int, help="How many epochs to train model on data collected from each round")
  parser.add_argument("--eval-games", default=10, type=positive_int, help="Number of evaluation games to play between newly trained model and previous best model")
  parser.add_argument("--win-threshold", default=0.55, type=constrained_float, help="Fraction of evaluation games that newly trained model must win to replace previous best model")
  parser.add_argument("--checkpoint-dir", required=True, type=valid_checkpoint_dir, help="Directory for model checkpoints (if checkpoints already exist in this directory, the trainer will use them as a starting point)")

  train_args = parser.add_argument_group("train_args")
  train_args.add_argument("--batch-size", default=10, type=positive_int, help="Batch size for training neural network")
  train_args.add_argument("--lr", default=3e-4, type=float, help="Learning rate for training neural neural network")

  game_args = parser.add_argument_group("game_args")
  game_args.add_argument("--puct-iters", default=500, type=positive_int, help="Number of PUCT simulations to execute for each move")
  game_args.add_argument("--c-puct", default=1, type=float, help="PUCT exploration coefficient")
  game_args.add_argument("--temp", default=1, type=float, help="PUCT exploration temperature")

  return parser.parse_args()

def parse_args_single_player():
  parser = argparse.ArgumentParser()
  parser.add_argument("--opponent", type=valid_opponent, help="Opponent to play against. One of: Random, AlphaBeta, PUCT, AlphaZero")
  parser.add_argument("--puct-iters", type=positive_int, help="Number of PUCT simulations to execute for each move (only valid for PUCT or AlphaZero opponents).")
  
  alpha_beta_args = parser.add_argument_group("alpha_beta_args")
  alpha_beta_args.add_argument("--depth", type=positive_int, help="Max depth of alpha-beta search (only valid for AlphaBeta opponent).")

  return parser.parse_args()