class ResignCounter(object):
    """description of class"""
    def __init__(self):
      self.allowed_resigns = 0
      self.disallowed_resigns = 0
      self.false_resigns = 0

    def update(self, outcome, resignations, resign_allowed):
      if (resign_allowed and (True in resignations.values())):
        self.allowed_resigns += 1
      elif (not resign_allowed):
        self.disallowed_resigns += sum(resignations.values())
        if (outcome == 0 or resignations[outcome]):
          self.false_resigns += 1

    def print_stats(self):
      print(f"\nTotal Resigns: {self.allowed_resigns}")
      frac_false_resigns = 0 if self.disallowed_resigns == 0 \
        else float(self.false_resigns) / float(self.disallowed_resigns)
      print(f"Proportion of false-positive resigns: {frac_false_resigns}")


