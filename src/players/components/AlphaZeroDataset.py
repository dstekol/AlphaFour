from torch.utils.data import Dataset
import torch
import random

class AlphaZeroDataset(Dataset):
    def __init__(self, game_trajectories, samples_per_game):
      self.data = []
      if (samples_per_game <= 0 or (samples_per_game > 1 and samples_per_game != int(samples_per_game))):
        raise ValueError("samples_per_game must be None, positive integer, or fraction in range (0, 1)")
      for game_trajectory in game_trajectories:
        self.data.append(self._process_trajectory(game_trajectory, samples_per_game))

    def _process_trajectory(self, game_trajectory, samples_per_game):
      outcome, game_state_trajectory, action_score_trajectory = game_trajectory
      outcome = torch.tensor([outcome])
      game_data = []
      for i in range(len(game_trajectory)):
        x = torch.tensor(game_state_trajectory[i])
        y = torch.tensor(action_score_trajectory[i]) # TODO action consistency
        y = torch.cat((y, outcome), dim=0)
        game_data.append((x,y))
      if (samples_per_game is None):
        return game_data
      elif (samples_per_game < 1):
        num_samples = int(len(game_data) * samples_per_game)
        return random.sample(game_data, num_samples)
      else:
        return random.sample(game_data, min(int(samples_per_game), len(game_data)))

    def __getitem__(self, index):
      return self.data[index]

    def __len__(self):
      return len(self.data)




