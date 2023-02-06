import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import numpy as np

class AlphaZeroCNN(LightningModule):
  """
  CNN/Resnet model for predicting Connect Four policy and state values.
  """

  def __init__(self, lr, l2_reg, state_value_weight):
    """
    Args:
    lr: learning rate
    l2_reg: Strength of L2 regularization
    state_value_weight: Weight of state-value loss relative to policy-value loss. 
      0 means training loss is equal to state-value loss (policy-value loss has no impace), 1 means vice versa. 
      0.5 means state-value and policy-value losses are equally weighted.
    """

    super().__init__()
    self.save_hyperparameters("lr", "l2_reg", "state_value_weight")

    # backbone
    self.conv1 = nn.Conv2d(3, 80, kernel_size=4, padding="same")
    self.drop1 = nn.Dropout(p = 0.5)
    self.conv2 = nn.Conv2d(80, 80, kernel_size=4, padding="same")
    self.drop2 = nn.Dropout(p = 0.5)
    self.conv3 = nn.Conv2d(80, 80, kernel_size=4, padding="same")
    self.drop3 = nn.Dropout(p = 0.5)
    self.conv4 = nn.Conv2d(80, 80, kernel_size=4, padding="same")
    self.drop4 = nn.Dropout(p = 0.5)
    self.fc1 = nn.Linear(42*80, 100)

    # policy-value head
    self.fc_policy = nn.Linear(100, 7)

    # state-value head
    self.fc_value = nn.Linear(100, 1)

    # losses
    self.policy_criterion = torch.nn.CrossEntropyLoss()
    self.value_criterion = torch.nn.MSELoss()


  def forward(self, inp, apply_softmax=False):
    # backbone
    out1 = F.relu(self.conv1(inp))
    out1 = self.drop1(out1)
    out2 = F.relu(self.conv2(out1) + out1)
    out2 = self.drop2(out2)
    out3 = F.relu(self.conv3(out2) + out2)
    out3 = self.drop3(out3)
    out4 = F.relu(self.conv4(out3) + out3)
    out4 = self.drop4(out4)
    out5 = out4.reshape(-1, 42*80)
    out = F.relu(self.fc1(out5))

    # policy-value head
    action_vals = self.fc_policy(out)
    # optionally apply softmax (for use during inference, but not training)
    if (apply_softmax):
      action_vals = F.softmax(action_vals, dim=1)

    # state-value head
    state_val = torch.tanh(self.fc_value(out))

    return action_vals, state_val

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2_reg)

  def on_train_start(self):
    self.logger.log_hyperparams(self.hparams, architecture_dict(self, "hp/"))

  def training_step(self, train_batch, ind):
    # preprocessing
    board, vals = train_batch
    action_vals = vals[:, :-1]
    state_vals = vals[:, -1].unsqueeze(1)

    # forward pass
    pred_action_vals, pred_state_vals = self(board)

    # losses
    policy_loss = self.policy_criterion(pred_action_vals, action_vals)
    value_loss = self.value_criterion(pred_state_vals, state_vals)
    loss = (1 - self.hparams.state_value_weight) * policy_loss + self.hparams.state_value_weight * value_loss

    # logging
    self.log("train/loss", loss)
    self.log("train/policy_loss", policy_loss)
    self.log("train/value_loss", value_loss)
    return loss

  def validation_step(self, val_batch, ind):
    # preprocessing
    board, vals = val_batch
    action_vals = vals[:, :-1]
    state_vals = vals[:, -1].unsqueeze(1)

    # forward pass
    pred_action_vals, pred_state_vals = self(board)

    # losses
    policy_loss = self.policy_criterion(pred_action_vals, action_vals)
    value_loss = self.value_criterion(pred_state_vals, state_vals)
    loss = (1 - self.hparams.state_value_weight) * policy_loss + self.hparams.state_value_weight * value_loss

    # logging
    self.log("val/loss", loss)
    self.log("val/policy_loss", policy_loss)
    self.log("val/value_loss", value_loss)

def architecture_dict(model, prefix):
  """
  Helper function for logging model architecture

  Args:
  model (torch.Module): neural network model
  prefix (String): logging prefix (for grouping in Tensorboard).

  Returns:
  Dictionary mapping layer name to output width.
  """
  widths = {}
  for name, child in model.named_children():
    name = prefix + name
    if isinstance(child, nn.Sequential):
      widths.update(get_layer_widths(child))
    elif (isinstance(child, nn.Conv2d)):
      widths[name] = child.out_channels
    elif (isinstance(child, nn.Linear)):
      widths[name] = child.out_features
  return widths


