from src.players.ConnectFourPlayer import ConnectFourPlayer
from src.ConnectFour import ConnectFour
import gym
import pettingzoo
import supersuit as ss
import numpy as np
import stable_baselines3 as sbs
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

class ReinforcePlayer(ConnectFourPlayer):
    """description of class"""
    def __init__(self, model_path, env):
      self.model = sbs.PPO.load(model_path, env)
      self.env = env

    def pick_move(self, game):
      pass

def apply_wrappers(env):
  env = pettingzoo.utils.wrappers.CaptureStdoutWrapper(env)
  #env = pettingzoo.utils.wrappers.AssertOutOfBoundsWrapper(env)
  env = pettingzoo.utils.wrappers.OrderEnforcingWrapper(env)
  return env

class ConnectFourEnv(pettingzoo.AECEnv):
  metadata = {'render.modes': ['human'], "name": "connect_four"}

  def __init__(self, obs_format):
    self.possible_agents = [-1, 1]
    self.action_spaces = gym.spaces.Discrete(7)
    self.obs_format = obs_format
    if (obs_format == "discrete"):
      self.observation_spaces = gym.spaces.MultiDiscrete([[3]*7]*6)
    elif (obs_format == "box"):
      self.observation_spaces = gym.spaces.Box(low=1, high=3, shape=(2, 3))
    else:
      raise ValueError(f"obs_format must be \"discrete\" or \"box\" but was {self.obs_format}")

  def render(self, mode="human"):
    self.game.print_board()

  def observe(self, agent):
    if (self.obs_format == "discrete"):
      return self._observe_discrete()
    elif (self.obs_format == "box"):
      return self._observe_box()
    else:
      raise ValueError(f"obs_format must be \"discrete\" or \"box\" but was {self.obs_format}")

  def _observe_discrete(self):
    convert_func = np.vectorize(lambda elt: 2 if elt == -1 else elt)
    return convert_func(self.game.board)

  def _observe_box(self):
    return self.game.board.copy()

  def close(self):
    return

  def reset(self):
    self.agents = self.possible_agents[:]
    self.rewards = {agent: 0 for agent in self.agents}
    self._cumulative_rewards = {agent: 0 for agent in self.agents}
    self.dones = {agent: False for agent in self.agents}
    self.infos = {agent: {} for agent in self.agents}
    self.game = ConnectFour()
    self._agent_selector = agent_selector(self.agents)
    self.agent_selection = self._agent_selector.next()

  def step(self, action):

    agent = self.agent_selection

    if (self.dones[agent]):
      return self._was_done_step(action)

    self.game.perform_move(action)
    
    is_over, winner = self.game.is_over()
    if (is_over):
      for player in self.agents:
        self.dones[player] = True
      if (winner != 0):
        self.rewards[winner] = 1
        self.rewards[winner * -1] = -1

    self.agent_selection = self._agent_selector.next()



env = ConnectFourEnv(obs_format="discrete")
env = apply_wrappers(env)
vec_env = ss.pettingzoo_env_to_vec_env_v0(env)
#multi_vec_env = ss.concat_vec_envs_v0(env, 8, num_cpus=2, base_class="stable_baselines3")
#model = sbs.PPO("MlpPolicy", multi_vec_env,  verbose=2)

#model.learn(total_timesteps=20000)
#model.save("ppo_mlp_discrete.model")



env.reset()
vec_env
for agent in env.agent_iter():
  observation, reward, done, info = env.last()
  action = int(input("action"))
  env.step(action)
  env.render()





