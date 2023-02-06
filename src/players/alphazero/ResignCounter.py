class ResignCounter(object):
    """Helper class for tracking number of resignations.
    This helps ensure the proportion of false-positive resignations is acceptable low."""

    def __init__(self):
      self.allowed_resigns = 0
      self.disallowed_resigns = 0
      self.false_resigns = 0

    def update(self, outcome, resignations, resign_allowed):
      """
      Updates resignation statistics upon conclusion of a particular game.

      Args:
      outcome (Integer): winner of game (1 for player 1, -1 for player 2, 0 if tie).
      resignations (Dict[Integer, Boolean]): mapping from player index (1 for player 1, -1 for player 2) to whether the corresponding player attempted to resign during the game.
      resign_allowed (Boolean): whether resignation was permitted (resignation is sometimes randomly disabled to accumulate false-positive statistics).
      """
      if (resign_allowed and (True in resignations.values())):
        self.allowed_resigns += 1
      elif (not resign_allowed):
        self.disallowed_resigns += sum(resignations.values())
        if (outcome == 0 or resignations[outcome]):
          self.false_resigns += 1

    def get_stats(self):
      """
      Returns:
      Dict[String, Float]: dictionary of resignation statistics (total number of resignations, number of false-positive resignations)
      """
      frac_false_resigns = 0 if self.disallowed_resigns == 0 \
        else float(self.false_resigns) / float(self.disallowed_resigns)
      return {"Total Resigns": self.allowed_resigns, "False-pos resigns": frac_false_resigns}


