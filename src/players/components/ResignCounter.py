class ResignCounter(object):
    """description of class"""
    def __init__(self):
      self.allowed_resigns = 0
      self.disallowed_resigns = 0
      self.false_resigns = 0

    def inc_allowed_resigns(self):
      self.allowed_resigns += 1

    def inc_false_resigns(self):
      self.false_resigns += 1

    def inc_disallowed_resigns(self):
      self.disallowed_resigns += 1

    def print_stats(self):
      print(f"\nTotal Resigns: {self.allowed_resigns}")
      print(f"Proportion of false-positive resigns: {float(false_resigns = 0) / float(self.disallowed_resigns)}")


