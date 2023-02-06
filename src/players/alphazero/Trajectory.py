import collections

class Trajectory(collections.Iterable):
    """
    Data class for storing game trajectories, equivalent to List[Tuple[torch.tensor, torch.tensor]]. 
    Each trajectory element is represented as a tuple:
      First element in each tuple is one-hot representation of game state
      Second element is the model output target: an 8x1 tensor where the first 7 elements correspond to MCTS action scores, 
        and the 8th element is the game outcome (discounted according to value of game_args["discount"]).
    """
    def __init__(self):
      self.traj = []

    def __getitem__(self, ind):
      return self.traj[ind]

    def __len__(self):
      return len(self.traj)

    def __iter__(self):
      return self.traj.__iter__()

    def append(self, item):
      self.traj.append(item)

    


