from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from src.players.alphazero.AlphaZeroNets import AlphaZeroCNN
from src.players.alphazero.AlphaZeroDataset import AlphaZeroDataset
from parse_utils import copy_args
import numpy as np
import math
import warnings

def create_datasets(game_trajectories, samples_per_game, flip_prob, validation_games):
  """
  Creates AlphaZeroDatasets for training and validation, based on the provided game_trajectories.
  Extracts the number of samples from each trajectory specified by samples_per_game

  Args:
  game_trajectories (List[Trajectory]): List representing set of game trajectories. 
    Each trajectory is represented as a list of state-outcome tuples representing the game trajectory.
    First element in each tuple is one-hot representation of game state. 
    Second element is the model output target: an 8x1 tensor where the first 7 elements correspond to MCTS action scores, 
    and the 8th element is the game outcome (discounted according to value of game_args["discount"]).
  samples_per_game (Float): The number of state-outcome pairs to use for training from each game trajectory. 
        If None, all data will be used. If in range (0,1), corresponding fraction of trajectories will be used. 
        If integer greater than or equal to 1, corresponding number of games per trajectory will be used.
  flip_prob (Float): Number in range (0,1) controlling the probability with which the random flip transformation is applied (ex. if flip_prob=0.5, half of samples will be flipped).
  validation_games: The number of  to use for training from each game trajectory.
        If None, all data will be used. If in range (0,1), corresponding fraction of trajectories will be used. 
        If integer greater than or equal to 1, corresponding number of games per trajectory will be used.

  Returns:
  train_data (AlphaZeroDataset)
  val_data (AlphaZeroDataset)
  """

  # compute number of game trajectories to hold out for validation
  if (validation_games is None):
    validation_games = 0
  elif (validation_games < 1):
    validation_games = int(len(game_trajectories) * validation_games)
  else: 
    validation_games = min(len(game_trajectories), validation_games)

  # perform data split
  data_inds = np.random.permutation(len(game_trajectories))
  val_inds = data_inds[:validation_games]
  train_inds = data_inds[validation_games:]

  train_trajectories = [game_trajectories[i] for i in train_inds] 
  val_trajectories = [game_trajectories[i] for i in val_inds] 

  train_data = AlphaZeroDataset(train_trajectories, samples_per_game, flip_prob)
  val_data = AlphaZeroDataset(val_trajectories, samples_per_game, flip_prob)
  return train_data, val_data

def init_model(net_args, device, source_model=None):
  """
  Initializes an AlphaZeroCNN model on the specified device, with arguments specified by net_args.
  If source_model is specified, the state_dict are copied from the source model.

  Args:
  net_args (Dict[String, Float]): subset of command line arguments containing values for the following items:
    "lr": learning rate
    "l2_reg": L2 regularization strength
    "value_weight": Float in range [0,1] controlling strength of state-value loss relative to policy-value loss
      (0 means only policy-value is ussed, 1 means only state-value loss is used, 0.5 means state-value and policy-value loss are equally weighted).
  device (String): "cpu" or "cuda"
  source_model (Optional[AlphaZeroCNN]): TODO

  Returns:
  model (AlphaZeroCNN)
  """
  net_arg_names = ["lr", "l2_reg", "state_value_weight"]
  net_args = copy_args(net_args, net_arg_names)
  model = AlphaZeroCNN(**net_args)
  if (source_model is not None):
    model.load_state_dict(source_model.state_dict())
  return model.to(device)

def train_model_single_version(model, 
                        train_loader, 
                        val_loader, 
                        round_ind, 
                        net_args, 
                        log_dir, 
                        device):
  """
  Performs a training run for given model

  model (pytorch_lightning.LightningModule): model to train
  train_loader (DataLoader): training data loader
  val_loader (DataLoader): validation data loader
  round_ind (Integer): index of current self-play round
  net_args (Dict[String, Any]): dictionary of command line argument relevant to training network.
    Must contain keys: patience, max_epochs, min_epochs
  log_dir (Path): path to tensorboard logging directory. Should be "logs" subdirectory of base logging directory (base_dir)
  device (String): "cpu" or "cuda"

  Returns:
  pytorch_lightning.Trainer: Trainer object corresponding to trained model (needed to have reference to trainer's save_checkpoint() method)
  """

  # initialize trainer
  name = f"round_{str(round_ind)}"
  tb_logger = TensorBoardLogger(save_dir=log_dir, name=name, default_hp_metric=False)
  if (net_args["patience"] is not None):
    early_stop_callback = EarlyStopping(monitor="val/loss", 
                                        mode="min", 
                                        patience=net_args["patience"])
    callbacks = [early_stop_callback]
  else:
    callbacks = []
  trainer = Trainer(max_epochs=net_args["max_epochs"], 
                       min_epochs=net_args["min_epochs"],
                       enable_checkpointing=False,
                       accelerator=("cpu" if device=="cpu" else "gpu"), 
                       devices=1,
                       logger=tb_logger,
                       val_check_interval=0.2,
                       enable_model_summary = False,
                       callbacks=callbacks
                       )

  # perform training run
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    trainer.fit(model, train_loader, val_loader)
  return trainer

def train_model_multi_version(model, 
                game_trajectories, 
                dataset_args, 
                round_ind, 
                net_args, 
                base_dir, 
                device):
  """
  Will train mutliple versions of given model (number of versions dictated by net_args["train_attempts"])
  and return the model with the highest validation loss (a random subset of trajectories will be held out for computing validation loss).

  Args:
  model (pytorch_lightning.LightningModule): model to train
  game_trajectories (List[Trajectory]): list of game trajectories to be used for training
  dataset_args (Dict[String, Any]): dictionary of command line arguments relevant to creating datasets.
    Must contain: samples_per_game, flip_prob, validation_games
  round_ind (Integer): index of current self-play round.
  net_args (Dict[String, Any]): dictionary of command line arguments relevant to training neural net model.
    Must contain: train_attempts, batch_size, patience, max_epochs, min_epochs
  base_dir (Path): base logging directory
  device (String): "cpu" or "cuda"

  Returns:
  model (pytorch_lightning.LightningModule): trained model
  save_checkpoint (Callable): reference to save_checkpoint() method, to be used to save trained model
  """
  model.train()
  log_dir = base_dir / "logs"
  best_val_loss = math.inf
  best_trainer = None
  best_model = None
  for i in range(net_args["train_attempts"]):
    # create model copy
    model_copy = init_model(net_args, device, model)
    model_copy.train()

    # initialize data loaders
    train_data, val_data = create_datasets(game_trajectories, **dataset_args)
    train_loader = DataLoader(train_data, 
                              shuffle=True, 
                              batch_size=net_args["batch_size"])
    val_loader = DataLoader(val_data, shuffle=False, batch_size=net_args["batch_size"])

    # train model
    trainer = train_model_single_version(model_copy, 
                                  train_loader, 
                                  val_loader, 
                                  round_ind, 
                                  net_args, 
                                  log_dir, 
                                  device)
    # update best model so far according to validation loss
    if (trainer.logged_metrics["val/loss"] < best_val_loss):
      best_val_loss = trainer.logged_metrics["val/loss"]
      best_model = model_copy
      best_trainer = trainer
    else:
      model_copy.cpu()
  return best_model.to(device), best_trainer.save_checkpoint
