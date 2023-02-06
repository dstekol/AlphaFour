import argparse
from pathlib import Path
import os
from enum import Enum

class AgentType(Enum):
  RANDOM = "random"
  ALPHABETA = "alphabeta"
  MCTS = "mcts"
  ALPHAZERO = "alphazero"

def positive_int(arg):
  try:
    i = int(arg)
    if (i <= 0):
      raise ValueError()
  except Exception as e:
    raise argparse.ArgumentTypeError("Must be a positive integer")
  return i

def nonnegative_int(arg):
  try:
    i = int(arg)
    if (i < 0):
      raise ValueError()
  except Exception as e:
    raise argparse.ArgumentTypeError("Must be a positive integer")
  return i

def constrained_float(arg):
  try:
    f = float(arg)
    if (f < 0 or f > 1):
      raise ValueError()
  except Exception as e:
    raise argparse.ArgumentTypeError("Must be a floating point number in range [0,1]")
  return f

def path_arg(arg):
  return Path(arg)

def str_to_bool(arg):
  if (str(arg).upper() == "FALSE"):
    return False
  elif (str(arg).upper() == "TRUE"):
    return True
  raise argparse.ArgumentTypeError("Must be either True or False")

def valid_opponent(arg):
  if (arg not in [e.value for e in AgentType]):
    raise argparse.ArgumentTypeError(f"Must be one of {str([e.value for e in AgentType])}")
  return arg
