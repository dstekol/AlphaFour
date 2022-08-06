from torch.utils.data import Dataset
import torch
import random

class AlphaZeroDataset(Dataset):
    def __init__(self, game_trajectories, samples_per_game=None, flip_prob=0.5):
      self.data = []
      self.flip_prob = flip_prob
      for trajectory in game_trajectories:
        num_samples = self._num_samples(samples_per_game, trajectory)
        trajectory_samples = random.choices(trajectory, k=min(len(trajectory), num_samples))
        self.data.extend(trajectory_samples)

    def _num_samples(self, samples_per_game, trajectory):
      if (samples_per_game is None):
        return len(trajectory)
      elif (0 <= samples_per_game < 1):
        return int(len(trajectory) * samples_per_game)
      else:
        return int(samples_per_game)
    
    def _random_flip(self, item):
      if (random.random() < self.flip_prob):
        x = torch.flip(item[0], [2])
        y = item[1].clone()
        y[:-1] = torch.flip(y[:-1], [0])
        return (x, y)
      else:
        return item

    def __getitem__(self, index):
      return self._random_flip(self.data[index])

    def __len__(self):
      return len(self.data)




