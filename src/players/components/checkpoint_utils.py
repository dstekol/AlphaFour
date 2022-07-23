import pytorch_lightning as pl
import numpy as np
import random
import os
import re
from src.players.components.AlphaZeroNets import AlphaZeroFCN

def get_ordered_checkpoints(checkpoint_dir):
  checkpoint_pattern = re.compile("[0-9]+\.ckpt")
  checkpoint_filter = lambda f: os.path.isfile(checkpoint_dir / f) and checkpoint_pattern.match(f)
  files = list(filter(checkpoint_filter, os.listdir(checkpoint_dir)))
  files.sort(key = lambda elt: int(elt.rstrip(".ckpt")))
  return files

def get_random_opponent(checkpoint_dir):
  ordered_checkpoints = get_ordered_checkpoints(checkpoint_dir)
  if (len(ordered_checkpoints) == 0):
    return None, ""
  if (len(ordered_checkpoints) > 1):
    ordered_checkpoints = ordered_checkpoints[:-1]
  checkpoint = random.choice(ordered_checkpoints)
  checkpoint_path = checkpoint_dir / checkpoint
  return AlphaZeroFCN.load_from_checkpoint(checkpoint_path), checkpoint

def get_latest_model(checkpoint_dir):
  ordered_checkpoints = get_ordered_checkpoints(checkpoint_dir)
  if (len(ordered_checkpoints) == 0):
    return None
  checkpoint_path = checkpoint_dir / ordered_checkpoints[-1]
  return AlphaZeroFCN.load_from_checkpoint(checkpoint_path)

def save_model(checkpoint_dir, trainer):
  ordered_checkpoints = get_ordered_checkpoints(checkpoint_dir)
  if (len(ordered_checkpoints)==0):
    max_checkpoint_num = -1
  else:
    max_checkpoint_num = int(ordered_checkpoints[-1].rstrip(".ckpt"))
  new_filename = str(max_checkpoint_num + 1) + ".ckpt"
  checkpoint_path = checkpoint_dir / new_filename
  trainer.save_checkpoint(checkpoint_path)