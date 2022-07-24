import threading
import torch
import numpy as np

class EvalBuffer:
  def __init__(self, max_buffer_size, max_wait_time):
    self.models = []
    self.input_buffers = dict()
    self.condition_buffers = dict()
    self.enqueue_times = dict()
    self.input_buffer_ids = dict()
    self.processed = dict()
    self.max_buffer_size = max_buffer_size
    self.enqueue_condition = threading.Condition()
    self.max_wait_time = max_wait_time
    self.lock = threading.Lock()
    self.buffer_id_counter = 0

  def get_enqueue_condition(self):
    return self.enqueue_condition

  def _get_buffer_id(self):
    self.buffer_id_counter += 1
    return self.buffer_id_counter - 1

  def get_max_wait_time(self):
    return self.max_wait_time

  def register_model(self, model):
    with self.lock:
      if (model in self.models):
        return
      self.models.append(model)
      model_ind = len(self.models)
      self.input_buffers[model_ind] = []
      self.condition_buffers[model_ind] = []
      self.input_buffer_ids = []
      self.enqueue_times[model_ind] = None

  def deregister_model(self, model):
    with self.lock:
      if (not model in self.models):
        return
      model_ind = self.models.index(model)
      self.models[self.models] = None
      self.input_buffers[model_ind] = []
      self.condition_buffers[model_ind] = []
      self.input_buffer_ids = []
      self.enqueue_times[model_ind] = None

  def queue_inputs(self, model, inputs):
    with self.lock:
      model_ind = self.models.index(model)
      condition = thread.Condition()
      buffer_ids = [self._get_buffer_id() for i in range(len(inps))]
      current_time = int(time.time() * 1000)

      self.condition_buffers[model_ind].append(condition)
      self.enqueue_times[model_ind] = current_time
      self.input_buffers[model_ind].extend(inputs)
      self.input_buffer_ids[model_ind].extend(buffer_ids)
    with self.enqueue_condition:
      self.enqueue_condition.notify()
    with condition:
      condition.wait()
    outputs = []
    for buffer_id in buffer_ids:
      outputs.append(self.processed[buffer_id])
      del self.processed[buffer_id]
    return outputs

  def flush(self):
    with self.lock:
      for model_ind, model in enumerate(self.models):
        enqueue_time = self.enqueue_times[model_ind]
        current_time = int(time.time() * 1000)
        buffer_len = len(self.input_buffers[model_ind])
        if (buffer_len >= self.max_buffer_size 
            or (enqueue_time is not None and current_time - enqueue_time >= self.max_wait_time)):
          input_buffer = self.input_buffers[model_ind]
          buffer_ids = self.input_buffer_ids[model_ind]
          conditions = self.condition_buffers[model_ind]

          x = torch.tensor(np.array(input_buffer)).to(model.device)
          actions_vals, state_vals = model(x)
          actions_vals = actions_vals.cpu().detach().numpy()
          state_vals = state_vals.cpu().detach().numpy()

          for i, buffer_id in enumerate(buffer_ids):
            self.processed[buffer_id] = (actions_vals[i], state_vals[i])
          
          self.input_buffers[model_ind] = []
          self.input_buffer_ids[model_ind] = []
          self.condition_buffers[model_ind] = []
          self.enqueue_times[model_ind] = None

          for condition in conditions:
            with condition:
              condition.notify()


def manage_buffer(eval_buffer):
  enqueue_condition = eval_buffer.get_enqueue_condition()
  max_wait_time = eval_buffer.get_max_wait_time()
  while (True):
    with enqueue_condition:
      enqueue_condition.wait(max_wait_time)
      eval_buffer.flush()
