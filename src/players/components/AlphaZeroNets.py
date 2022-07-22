import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class AlphaZeroFCN(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.fc1 = nn.Linear(7*6*3, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, 50)
    self.fc_policy = nn.Linear(50, 7)
    self.fc_value = nn.Linear(50, 1)
    self.cfg = cfg

  def forward(self, board):
    inp = torch.tensor(board, dtype=torch.float, device=self.fc1.device)
    inp = inp.reshape(-1, 7*6*3)
    out = F.relu(self.fc1(inp))
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))
    action_vals = self.fc_policy(out)
    state_val = self.fc_value(out)
    return action_vals, state_val

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=cfg["lr"]) # TODO hyperparams

  def training_step(self, train_batch, ind):
    board, vals = train_batch
    norm_vals = (vals + 1) / 2
    norm_action_vals, norm_state_vals = norm_vals[:, :-1], norm_vals[:, -1]
    pred_action_vals, pred_state_vals = self(board)
    policy_loss = F.binary_cross_entropy_with_logits(pred_action_vals, norm_action_vals)
    value_loss = F.binary_cross_entropy_with_logits(pred_state_vals, norm_state_vals)
    loss = cfg["policy_weight"] * policy_loss + cfg["value_weight"] * value_loss
    return loss



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
    inp = torch.tensor(board, dtype=torch.float, device=self.fc1.device)
    out = F.relu(self.conv1(inp))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))
    out = out.reshape(-1, 12*10)
    out = F.relu(self.fc1(out))
    action_vals = F.tanh(out)
    state_val = F.tanh(out)
    return action_vals, state_val

