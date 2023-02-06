import threading
import torch
import numpy as np
import time

class BufferedModelWrapper:
  """
  A wrapper around around a neural network model (see AlphaZeroNets) which 
  buffers evaluation requests for more efficient use of the GPU. The buffer is 
  flushed when either the max_wait_time elapses or the max_buffer_size is reached.
  """

  def __init__(self, model, max_buffer_size, max_wait_time):
    """
    Args:
    model (Union[torch.nn.Module, pytorch_lightning.LightningModule]): neural net object to be used for inference
    max_buffer_size (Integer): the maximum amount of inference requests to buffer before flushing the queue
    max_wait_time (Float): the maximum amount of time (in seconds) to wait before flushing the queue. 
      Resets whenever new inference request is made.
    """

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
    """
    Asynchronously monitors queue from separate thread,
    and flushes whenever buffer size or wait time is exceeded.
    """

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
    """
    Shuts down queue and corresponding monitoring thread.
    """

    self.active = False
    with self.enqueue_condition:
      self.enqueue_condition.notify()
    self.manager_thread.join()

  def _generate_buffer_id(self):
    """
    Generates a new id for the current queue (resets on each flush)
    """
    self.buffer_id_counter += 1
    return self.buffer_id_counter - 1

  def enqueue(self, inputs):
    """
    Adds an input to the evaluation queue, blocking until queue is flushed.

    Args:
    inputs (Union[np.ndarray, torch.tensor]): tensor of shape (n, 3, 6, 7) representing 
      one-hot board states to be evaluated by neural net model

    Returns:
    action_vals (np.ndarray): array of shape (n, 7) representing action values of each column
    state_vals (np.ndarray): array of shape (n,) representing state values of each input
    """

    # enqueues inputs, request time, and wait condition (to be notified when processing finished)
    with self.lock:
      condition = threading.Condition()
      buffer_ids = [self._generate_buffer_id() for i in range(len(inputs))]
      self.condition_buffer.append(condition)
      self.enqueue_time = int(time.time() * 1000)
      self.input_buffer.extend(inputs)
      self.input_buffer_ids.extend(buffer_ids)
    
    # notifies manager thread of new input
    with self.enqueue_condition:
      self.enqueue_condition.notify()
    
    # waits until processing finished
    with condition:
      if (condition in self.condition_buffer):
        condition.wait()
    
    # extracts outputs from output queue
    outputs = []
    for buffer_id in buffer_ids:
      outputs.append(self.processed[buffer_id]) 
      del self.processed[buffer_id]
    return outputs

  def _flush(self):
    """
    Flushes the current evaluation queue.
    """
    with self.lock:
      # performs neural net forward pass
      with torch.no_grad():
        x = torch.tensor(np.array(self.input_buffer), dtype=torch.float32, device=self.model.device)
        actions_vals, state_vals = self.model(x, apply_softmax=True)
        actions_vals = actions_vals.cpu().detach().numpy()
        state_vals = state_vals.cpu().detach().numpy()

      # saves outputs to output queue, with corresponding buffer ids as keys
      for i, buffer_id in enumerate(self.input_buffer_ids):
        self.processed[buffer_id] = (actions_vals[i], state_vals[i])
        
      conditions = self.condition_buffer

      # resets buffers
      self.input_buffer = []
      self.input_buffer_ids = []
      self.condition_buffer = []
      self.enqueue_time = None

      # wakes all waiting threads
      for condition in conditions:
        with condition:
          condition.notify()
