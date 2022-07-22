import pytorch_lightning as pl
import numpy as np
import random
import os
import re
from src.players.components.AlphaZeroNets import AlphaZeroFCN

def get_ordered_checkpoints(checkpoint_dir):
  checkpoint_pattern = re.compile("[0-9]+\.ckpt")
  checkpoint_filter = lambda f: os.isfile(f) and checkpoint_pattern.matches(f)
  files = list(filter(checkpoint_filter, os.listdir(checkpoint_dir)))
  files.sort(key = lambda elt: int(elt.rstrip(".ckpt")))
  return files

def get_random_opponent(checkpoint_dir):
  ordered_checkpoints = get_ordered_checkpoints(checkpoint_dir)
  if (len(ordered_checkpoints) == 0):
    return None
  if (len(ordered_checkpoints) > 1):
    ordered_checkpoints = ordered_checkpoints[:-1]
  checkpoint_path = checkpoint_dir / random.choice(ordered_checkpoints)
  return AlphaZeroFCN.load_from_checkpoint(checkpoint_path)

def get_latest_model(checkpoint_dir):
  ordered_checkpoints = get_ordered_checkpoints(checkpoint_dir)
  if (len(ordered_checkpoints) == 0):
    return None
  checkpoint_path = checkpoint_dir / ordered_checkpoints[-1]
  return AlphaZeroFCN.load_from_checkpoint(checkpoint_path)

def save_model(checkpoint_dir, trainer):
  ordered_checkpoints = get_ordered_checkpoints(checkpoint_dir)
  filename = str(len(ordered_checkpoints) + 1) + ".ckpt"
  checkpoint_path = checkpoint_dir / filename
  trainer.save_checkpoint(checkpoint_path)