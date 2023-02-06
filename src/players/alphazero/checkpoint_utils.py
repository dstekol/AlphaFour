import pytorch_lightning as pl
import numpy as np
import random
import os
import re
import pickle as pkl
from src.players.alphazero.AlphaZeroNets import AlphaZeroCNN

def init_dirs(base_dir):
  """
  Initializes base logging directory, with subdirectory "logs" for Tensorboard logs, 
    "models" for model checkpoints, and "games" for saved self-play games.

  Args: 
  base_dir (Path): path to base logging directory (will be created if does not yet exist).
  """
  for dir in [base_dir, 
              base_dir / "logs", 
              base_dir / "models", 
              base_dir / "games"]:
    dir.mkdir(parents=True, exist_ok=True)

def file_num(file_name, ext):
  """
  Helper function for extracting number from filenames of format [num].[ext] (for instance 11.pkl).

  Args:
  file_name (String): file name, with extension (ex. "11.pkl")
  ext (String): file extension (ex. ".pkl")
  """
  return int(file_name.rstrip(ext))

def get_ordered_checkpoints(dir, ext):
  """
  Gets an ordered list of existing model checkpoints.

  Args:
  dir (Path): path to directory where model checkpoints are saved
  ext (String): checkpoint file extension (ex. ".ckpt")

  Returns:
  List[Integer] corresponding to training rounds at which model checkpoints were saved.
    Checkpoint files are named with the convention [round_num].ckpt (ex. "1.ckpt")
  """
  checkpoint_pattern = re.compile("[0-9]+\\" + ext)
  checkpoint_filter = lambda f: os.path.isfile(dir / f) and checkpoint_pattern.match(f)
  files = list(filter(checkpoint_filter, os.listdir(dir)))
  files.sort(key = lambda f: file_num(f, ext))
  return files

def get_latest_model(base_dir, device):
  """
  Loads the most recently saved model checkpoint to the specified device.

  Args:
  base_dir (Path): the base logging directory (with model checkpoints in the "models" subdirectory).
  device (String): "cpu" to load the model onto the CPU, or "cuda" to load onto GPU.

  Returns:
  AlphaZeroCNN: neural net object instance loaded from checkpoint.
  """

  checkpoint_dir = base_dir / "models"
  ordered_checkpoints = get_ordered_checkpoints(checkpoint_dir, ".ckpt")
  if (len(ordered_checkpoints) == 0):
    return None, -1
  checkpoint_path = checkpoint_dir / ordered_checkpoints[-1]
  model = AlphaZeroCNN.load_from_checkpoint(checkpoint_path).to(device), \
    file_num(ordered_checkpoints[-1], ".ckpt")
  return model

def save_model(base_dir, save_checkpoint_handle, round):
  """
  Saves model checkpoint.

  Args:
  base_dir (Path): base logging directory. Model will be saved in [base_dir]/models subdirectory.
  save_checkpoint_handle (Callable): reference to save_checkpoint() method of PytorchLightning object.
  round (Number): current round of training. Model will be saved as [round].ckpt
  """
  checkpoint_dir = base_dir / "models"
  new_filename = str(round) + ".ckpt"
  checkpoint_path = checkpoint_dir / new_filename
  save_checkpoint_handle(checkpoint_path)

def load_game_trajectories(base_dir):
  """
  Loads saved game trajectories when restarting from checkpoint

  Args:
  base_dir (Path): base logging directory, with trajectores saved in the "games" subfolder

  Returns:
  List[Trajectory]: loaded trajectories
  """
  games_dir = base_dir / "games"
  ordered_games = get_ordered_checkpoints(games_dir, ".pkl")
  if (len(ordered_games) == 0):
    return [], -1
  else:
    games_file = ordered_games[-1]
    file_path = games_dir / games_file
    return pkl.load(open(file_path, "rb")), file_num(games_file, ".pkl")

def save_game_trajectories(base_dir, trajectories, round):
  """
  Saves game trajectories in "model" subfolder of base_dir.
  Named according to the convention [round].pkl

  Args:
  base_dir (Path): base logging directory
  trajectories (List[Trajectory]): list of trajectories to save
  round (Integer): current training round
  """
  games_dir = base_dir / "games"
  new_filename = (str(round) + ".pkl")
  file_path = games_dir / new_filename
  pkl.dump(trajectories, open(file_path, "wb"))

def save_args(args):
  """
  Helper function for saving command line arg dictionary to "args.txt" file in base logging directory.

  Args:
  args (Dict[String, Any]): dictionary of command line arguments.
  """
  base_dir = args["base_dir"]
  s = str(args)
  s = "\n" + ",\n".join(s.split(","))
  filepath = base_dir / "info.txt"
  with filepath.open("a") as f:
    f.write(s)

def log_stats(base_dir, round, stats):
  """
  Helper function for saving training statistics to "info.txt" file in base logging directory.

  Args:
  base_dir (Path): path to base logging directory
  round (Integer): current training round
  state (Dict[String, Any]): dictionary of key-value pairs to log
  """
  filepath = base_dir / "info.txt"
  with filepath.open("a") as f:
    f.write(f"\n\nRound {round}:")
    for stat in stats:
      f.write(f"\n{stat}: {stats[stat]}")
