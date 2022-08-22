from torch import rand, zeros
from src.players.components.EvalBuffer import EvalBuffer
import threading
import numpy as np

class FakeModel:
  def __init__(self):
    self.device = "cuda"

  def __call__(self, inp, apply_softmax=False):
    return rand(inp.shape[0], 7, device=self.device), \
      rand(inp.shape[0], 1, device=self.device)

model = FakeModel()
buffer = EvalBuffer(model, 5, 5 * 1000)

enq = lambda buffer: print(buffer.enqueue([np.zeros((6, 7))]))

threads = []
for i in range(6):
  thread = threading.Thread(target=enq, args=[buffer])
  thread.start()
  threads.append(thread)

for thread in threads:
  thread.join()

buffer.close()


#import threading
#import multiprocessing as mp
#import time

#def f(k, l):
#    print("starting thread")
#    counter = 0
#    while (l.qsize() < k):
#      counter += 1
#      l.put(0)
#    while (not l.empty()):
#      l.get()
#    print(f"ending thread: {counter}")
    
    

#if __name__ == "__main__":
#  l = mp.Queue()

#  threads = []
#  for i in range(4):
#    thread = mp.Process(target=f, args=[1e5, l])
#    threads.append(thread)

#  print("starting")
#  s = time.time()
#  [thread.start() for thread in threads]
#  [thread.join() for thread in threads]
#  e = time.time()
#  print("done")
#  print(e - s)



