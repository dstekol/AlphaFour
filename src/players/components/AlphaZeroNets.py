import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class AlphaZeroFCN(pl.LightningModule):
  def __init__(self, **kwargs):
    super().__init__()
    self.save_hyperparameters("lr", "l2_reg", "policy_weight", "value_weight")
    self.fc1 = nn.Linear(7*6*3, 150)
    self.fc2 = nn.Linear(150, 150)
    self.fc_policy = nn.Linear(150, 7)
    self.fc_value = nn.Linear(150, 1)
    self.policy_criterion = torch.nn.CrossEntropyLoss()
    self.value_criterion = torch.nn.MSELoss()

  def forward(self, x, apply_softmax=False):
    x = x.reshape(-1, 7*6*3)
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    action_vals = self.fc_policy(out)
    if (apply_softmax):
      action_vals = F.softmax(action_vals, dim=1)
    state_val = torch.tanh(self.fc_value(out))
    return action_vals, state_val

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2_reg)

  def on_train_start(self):
    self.logger.log_hyperparams(self.hparams, architecture_dict(self, "hp/"))

  def training_step(self, train_batch, ind):
    board, vals = train_batch
    action_vals = vals[:, :-1]
    state_vals = vals[:, -1].unsqueeze(1)
    pred_action_vals, pred_state_vals = self(board)
    policy_loss = self.policy_criterion(pred_action_vals, action_vals)
    value_loss = self.value_criterion(pred_state_vals, state_vals)
    loss = self.hparams.policy_weight * policy_loss + self.hparams.value_weight * value_loss
    self.log("train/loss", loss)
    self.log("train/policy_loss", policy_loss)
    self.log("train/value_loss", value_loss)
    return loss

  def validation_step(self, val_batch, ind):
    board, vals = val_batch
    action_vals = vals[:, :-1]
    state_vals = vals[:, -1].unsqueeze(1)
    pred_action_vals, pred_state_vals = self(board)
    policy_loss = self.policy_criterion(pred_action_vals, action_vals)
    value_loss = self.value_criterion(pred_state_vals, state_vals)
    loss = self.hparams.policy_weight * policy_loss + self.hparams.value_weight * value_loss
    self.log("val/loss", loss)
    self.log("val/policy_loss", policy_loss)
    self.log("val/value_loss", value_loss)


class AlphaZeroCNN(pl.LightningModule):
  def __init__(self, **kwargs):
    super().__init__()
    self.save_hyperparameters("lr", "l2_reg", "policy_weight", "value_weight")
    self.conv1 = nn.Conv2d(3, 30, kernel_size=4, padding="valid")
    self.conv2 = nn.Conv2d(30, 40, kernel_size=4, padding="same")
    self.conv3 = nn.Conv2d(40, 40, kernel_size=3, padding="same")
    self.conv4 = nn.Conv2d(40, 40, kernel_size=3, padding="same")
    self.fc1 = nn.Linear(12*40, 50)
    self.fc_policy = nn.Linear(50, 7)
    self.fc_value = nn.Linear(50, 1)
    self.policy_criterion = torch.nn.CrossEntropyLoss()
    self.value_criterion = torch.nn.MSELoss()

  def forward(self, inp, apply_softmax=False):
    out = F.relu(self.conv1(inp))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))
    out = F.relu(self.conv4(out))
    out = out.reshape(-1, 12*40)
    out = F.relu(self.fc1(out))
    action_vals = self.fc_policy(out)
    if (apply_softmax):
      action_vals = F.softmax(action_vals, dim=1)
    state_val = torch.tanh(self.fc_value(out))
    return action_vals, state_val

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2_reg)

  def on_train_start(self):
    self.logger.log_hyperparams(self.hparams, architecture_dict(self, "hp/"))

  def training_step(self, train_batch, ind):
    board, vals = train_batch
    action_vals = vals[:, :-1]
    state_vals = vals[:, -1].unsqueeze(1)
    pred_action_vals, pred_state_vals = self(board)
    policy_loss = self.policy_criterion(pred_action_vals, action_vals)
    value_loss = self.value_criterion(pred_state_vals, state_vals)
    loss = self.hparams.policy_weight * policy_loss + self.hparams.value_weight * value_loss
    self.log("train/loss", loss)
    self.log("train/policy_loss", policy_loss)
    self.log("train/value_loss", value_loss)
    return loss

  def validation_step(self, val_batch, ind):
    board, vals = val_batch
    action_vals = vals[:, :-1]
    state_vals = vals[:, -1].unsqueeze(1)
    pred_action_vals, pred_state_vals = self(board)
    policy_loss = self.policy_criterion(pred_action_vals, action_vals)
    value_loss = self.value_criterion(pred_state_vals, state_vals)
    loss = self.hparams.policy_weight * policy_loss + self.hparams.value_weight * value_loss
    self.log("val/loss", loss)
    self.log("val/policy_loss", policy_loss)
    self.log("val/value_loss", value_loss)



def architecture_dict(model, prefix):
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


