from torch.utils.data import Dataset
import torch
import random

class AlphaZeroDataset(Dataset):
    """
    Custom Dataset class for Connect Four game trajectories.
    Includes trajectory subsampling and random-flip data augmentation.
    """

    def __init__(self, game_trajectories, samples_per_game=None, flip_prob=0.5):
      """
      Args:
      game_trajectories (List[Trajectory]): List of self-play trajectories.
      samples_per_game (Optional[Float]): The number of state-outcome pairs to use for training from each game trajectory. 
        If None, all data will be used. If in range (0,1), corresponding fraction of trajectories will be used. 
        If integer greater than or equal to 1, corresponding number of games per trajectory will be used.
      flip_prob (Optional[Float]): Float in range [0,1]. The probability with which the random flip transformation is applied (ex. if flip_prob=0.5, half of samples will be flipped).
      """
      self.data = []
      self.flip_prob = flip_prob
      for trajectory in game_trajectories:
        num_samples = self._num_samples(samples_per_game, trajectory)
        trajectory_samples = random.choices(trajectory, k=min(len(trajectory), num_samples))
        self.data.extend(trajectory_samples)

    def _num_samples(self, samples_per_game, trajectory):
      """
      Returns appropriate number of state-outcome pairs to sample from trajectory

      Args:
      samples_per_game (Float): The number of state-outcome pairs to use for training from each game trajectory. 
        If None, all data will be used. If in range (0,1), corresponding fraction of trajectories will be used. 
        If integer greater than or equal to 1, corresponding number of games per trajectory will be used.
      trajectory (Trajectory): Trajectory object containing list of (state, target_output) tuples.
      
      Returns: 
      Number of samples to use from given trajectory.
      """
      if (samples_per_game is None):
        return len(trajectory)
      elif (0 <= samples_per_game < 1):
        return int(len(trajectory) * samples_per_game)
      else:
        return int(samples_per_game)
    
    def _random_flip(self, item):
      """
      Applies horizontal flip to state-outcome pair (flips state tensor and policy section of target_output tensor),
        with probability self.flip_prob

      Args:
      item Tuple[torch.tensor, torch.tensor]: tuple of form (state, target_output) to be potentially flipped.
      """

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




