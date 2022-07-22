import argparse
from pathlib import Path
import os

def positive_int(arg):
  try:
    i = int(arg)
    if (i <= 0):
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

def valid_checkpoint_dir(arg):
  if (not os.path.isdir(arg)):
    argparse.ArgumentTypeError("Must be a valid directory")
  return Path(arg)

def valid_opponent(arg):
  pass # TODO
