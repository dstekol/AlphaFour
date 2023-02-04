import threading
import torch
import numpy as np
import time

class EvalBuffer:
  def __init__(self, model, max_buffer_size, max_wait_time):
    self.model = model
    self.input_buffer = []
    self.condition_buffer = []
    self.enqueue_time = None
    self.input_buffer_ids = []
    self.processed = dict()
    self.max_buffer_size = max_buffer_size
    self.enqueue_condition = threading.Condition()
    self.max_wait_time = max_wait_time
    self.lock = threading.RLock()
    self.buffer_id_counter = 0
    self.active = True
    self.manager_thread = threading.Thread(target = self._manage_buffer, args=[])
    self.manager_thread.start()
    

  def _manage_buffer(self):
    with self.enqueue_condition:
      while (self.active):
        self.enqueue_condition.wait(self.max_wait_time / 1000)
        current_time = int(time.time() * 1000)
        buffer_len = len(self.input_buffer)
        if (buffer_len >= self.max_buffer_size 
          or (self.enqueue_time is not None 
              and current_time - self.enqueue_time >= self.max_wait_time)):
          self._flush()

  def close(self):
    self.active = False
    with self.enqueue_condition:
      self.enqueue_condition.notify()
    self.manager_thread.join()

  def _get_buffer_id(self):
    self.buffer_id_counter += 1
    return self.buffer_id_counter - 1

  #def check_duplicates(self):
  #  inputs = self.input_buffer[:]
  #  s = set([item.tobytes() for item in inputs])
  #  print(f"dups: {len(inputs) - len(s)}")

  def enqueue(self, inputs):
    with self.lock:
      condition = threading.Condition()
      buffer_ids = [self._get_buffer_id() for i in range(len(inputs))]

      self.condition_buffer.append(condition)
      self.enqueue_time = int(time.time() * 1000)
      self.input_buffer.extend(inputs)
      self.input_buffer_ids.extend(buffer_ids)
    with self.enqueue_condition:
        self.enqueue_condition.notify()
    with condition:
      #print(f"waiting {threading.get_ident()}")
      if (condition in self.condition_buffer):
        condition.wait()
      #print(f"unwaiting {threading.get_ident()}")
    outputs = []
    for buffer_id in buffer_ids:
      outputs.append(self.processed[buffer_id]) 
      del self.processed[buffer_id]
    return outputs

  def _flush(self):
    with self.lock:
      with torch.no_grad():
        #self.check_duplicates()
        x = torch.tensor(np.array(self.input_buffer), dtype=torch.float32, device=self.model.device)
        actions_vals, state_vals = self.model(x, apply_softmax=True)
        actions_vals = actions_vals.cpu().detach().numpy()
        state_vals = state_vals.cpu().detach().numpy()

      for i, buffer_id in enumerate(self.input_buffer_ids):
        self.processed[buffer_id] = (actions_vals[i], state_vals[i])
        
      conditions = self.condition_buffer

      self.input_buffer = []
      self.input_buffer_ids = []
      self.condition_buffer = []
      self.enqueue_time = None

      for condition in conditions:
        with condition:
          condition.notify()
