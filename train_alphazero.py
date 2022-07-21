import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from src.players.ReinforcePlayer import ConnectFourEnv

cnct4_env = ConnectFourEnv()

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

#trainer.import_model("my_weights.h5")
