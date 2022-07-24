import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class AlphaZeroFCN(pl.LightningModule):
  def __init__(self, **kwargs):
    super().__init__()
    self.save_hyperparameters("lr", "l2_reg", "policy_weight", "value_weight")
    self.fc1 = nn.Linear(7*6*3, 100)
    #self.fc2 = nn.Linear(100, 100)
    self.fc2 = nn.Linear(100, 50)
    self.fc_policy = nn.Linear(50, 7)
    self.fc_value = nn.Linear(50, 1)

  def forward(self, x):
    x = x.reshape(-1, 7*6*3)
    out = F.relu(self.fc1(x))
    #out = F.relu(self.fc2(out))
    out = F.relu(self.fc2(out))
    action_vals = self.fc_policy(out)
    state_val = torch.tanh(self.fc_value(out))
    return action_vals, state_val

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2_reg)

  def training_step(self, train_batch, ind):
    board, vals = train_batch
    norm_vals = (vals + 1) / 2
    norm_action_vals = norm_vals[:, :-1]
    norm_state_vals = norm_vals[:, -1].unsqueeze(1)
    pred_action_vals, pred_state_vals = self(board)
    policy_loss = F.binary_cross_entropy_with_logits(pred_action_vals, norm_action_vals)
    value_loss = F.mse_loss(pred_state_vals, norm_state_vals)
    loss = self.hparams.policy_weight * policy_loss + self.hparams.value_weight * value_loss
    self.log("train/loss", loss)
    self.log("train/policy_loss", policy_loss)
    self.log("train/value_loss", value_loss)
    return loss

  def validation_step(self, val_batch, ind):
    board, vals = val_batch
    norm_vals = (vals + 1) / 2
    norm_action_vals = norm_vals[:, :-1]
    norm_state_vals = norm_vals[:, -1].unsqueeze(1)
    pred_action_vals, pred_state_vals = self(board)
    policy_loss = F.binary_cross_entropy_with_logits(pred_action_vals, norm_action_vals)
    value_loss = F.mse_loss(pred_state_vals, norm_state_vals)
    loss = self.hparams.policy_weight * policy_loss + self.hparams.value_weight * value_loss
    self.log("val/loss", loss)
    self.log("val/policy_loss", policy_loss)
    self.log("val/value_loss", value_loss)



class AlphaZeroCNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 20, kernel_size=4, padding=0)
    self.conv2 = nn.Conv2d(20, 30, kernel_size=4, padding_mode="same")
    self.conv3 = nn.Conv2d(30, 10, kernel_size=3, padding_mode="same")
    self.fc1 = nn.Linear(12*10, 50)
    self.fc_policy = nn.Linear(50, 7)
    self.fc_value = nn.Linear(50, 1)

  def forward(self, board):
    out = F.relu(self.conv1(inp))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))
    out = out.reshape(-1, 12*10)
    out = F.relu(self.fc1(out))
    action_vals = F.tanh(out)
    state_val = F.tanh(out)
    return action_vals, state_val

