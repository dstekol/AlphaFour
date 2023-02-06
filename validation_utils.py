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
  except ValueError:
    argparse.ArgumentTypeError("Must be a positive integer")
  return i

def nonnegative_int(arg):
  try:
    i = int(arg)
    if (i < 0):
      raise ValueError()
  except ValueError:
    argparse.ArgumentTypeError("Must be a positive integer")
  return i

def constrained_float(arg):
  try:
    f = float(arg)
    if (f < 0 or f > 1):
      raise ValueError()
  except ValueError:
    argparse.ArgumentTypeError("Must be a floating point number in range [0,1]")
  return f

def path_arg(arg):
  return Path(arg)

def valid_opponent(arg):
  if (arg not in [e.value for e in AgentType]):
    argparse.ArgumentTypeError(f"Must be one of {str([e.value for e in AgentType])}")
  return arg
