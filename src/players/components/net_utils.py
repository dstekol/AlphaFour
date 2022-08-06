from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from src.players.components.AlphaZeroNets import AlphaZeroFCN, AlphaZeroCNN
import numpy as np
import math

def init_model(train_args, device, source_model=None):
  net_arg_names = ["lr", "l2_reg", "value_weight"]
  net_args = {arg_name: train_args[arg_name] for arg_name in net_arg_names}
  model = AlphaZeroFCN(**net_args)
  if (source_model is not None):
    model.load_state_dict(source_model.state_dict())
  return model.to(device)

def train_model_attempt(model, train_loader, val_loader, round_ind, train_args, device):
  name = "round_" + str(round_ind)
  tb_logger = TensorBoardLogger(save_dir=train_args["log_dir"], name=name, default_hp_metric=False)
  early_stop_callback = EarlyStopping(monitor="val/loss", mode="min", patience=train_args["patience"])
  trainer = Trainer(max_epochs=train_args["epochs_per_round"], 
                       enable_checkpointing=False,
                       accelerator=("cpu" if device=="cpu" else "gpu"), 
                       devices=1,
                       #detect_anomaly=True,
                       logger=tb_logger,
                       val_check_interval=0.1,
                       log_every_n_steps=15,
                       #track_grad_norm=2,
                       #auto_lr_find=True,
                       enable_model_summary = False,
                       callbacks=[early_stop_callback]
                       )
  #trainer.tune(model, train_loader)
  trainer.fit(model, train_loader, val_loader)
  return trainer

def train_model(model, train_data, val_data, round_ind, train_args, device):
  model.train()
  train_loader = DataLoader(train_data, shuffle=True, batch_size=train_args["batch_size"])
  val_loader = DataLoader(val_data, shuffle=False, batch_size=train_args["batch_size"])
  best_val_loss = math.inf
  best_trainer = None
  best_model = None
  for i in range(train_args["train_attempts"]):
    model_copy = init_model(train_args, device, model)
    trainer = train_model_attempt(model_copy, train_loader, val_loader, round_ind, train_args, device)
    if (trainer.logged_metrics["val/loss"] < best_val_loss):
      best_val_loss = trainer.logged_metrics["val/loss"]
      best_model = model_copy
      best_trainer = trainer
  return best_model, best_trainer.save_checkpoint
