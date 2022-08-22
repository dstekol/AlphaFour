import torch
from pytorch_lightning import Trainer, Callback, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pickle as pkl
from src.ConnectFour import ConnectFour
from src.players.AlphaZeroPlayer import one_hot_state
from src.players.components.AlphaZeroNets import AlphaZeroFCN, AlphaZeroCNN
from src.players.components.AlphaZeroDataset import AlphaZeroDataset
from train_alpha_zero import create_datasets, train_model
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from itertools import product

#seed_everything(24)
#filename = "finished_game_data.pkl"
#trajectories = pkl.load(open(filename, "rb"))
#data = []
#for moves, scores, winner in tqdm(trajectories):
#  game = ConnectFour()
#  trajectory = []
#  for i, (move, action_scores) in enumerate(zip(moves, scores)):
#    state = one_hot_state(game).astype("int8")
#    player = 1 if i % 2 == 0 else -1
#    outcome = player * winner
#    target = np.concatenate((action_scores, [outcome]), axis=0)
#    trajectory.append((state, target))
#    game.perform_move(move)
#  data.append(trajectory)
#pkl.dump(data, open("processed_data.pkl", "wb"))
#seed_everything(20) # 1 high
def prep(data, samples):
  discount = 1 #0.96
  coeff = np.ones((8,))
  coeff[-1] = 1e5
  out = []
  for traj in data:
    traj = list(enumerate(reversed(traj)))
    traj_subset = traj #random.choices(traj,k=min(len(traj), samples))
    #out.append([(torch.tensor(elt[0], dtype=torch.float32), 
    #             torch.tensor(elt[1] * (coeff ** i), dtype=torch.float32)) 
    #              for i, elt in traj_subset])
    #out.append([(torch.tensor(elt[0], dtype=torch.float32), 
    #             torch.clip(torch.tensor(elt[1] * coeff, dtype=torch.float32), min=-1, max=1))
    #              for i, elt in traj_subset])
    out.append([(torch.tensor(elt[0], dtype=torch.float32), 
              torch.tensor(elt[1], dtype=torch.float32))
              for i, elt in traj_subset])
  return out
def viz(elt):
  state, target = elt
  board = (state.argmax(dim=0) - 1).numpy()
  game = ConnectFour(board)
  game.print_board()
  print(target[:7])
  print(target[7])
#short_data = pkl.load(open("a0_game_trajectories v3 (1500 @ 400).pkl", "rb"))
short_data = pkl.load(open("games_v7.pkl", "rb"))
#long_data = pkl.load(open("processed_data.pkl", "rb"))
#data = prep(long_data, 43)
print(len(short_data))
data = short_data[1500:]
#begin_data = [traj[:5] for traj in data]
#end_data = [traj[5:] for traj in data]

#data = long_data #[:4500]
#data = prep(data, 43)

def analyze(trajs):
  subopt_counter = 0
  diff_counter = 0
  move_counter = 0
  for traj in trajs:
    move_counter += len(traj) - 1
    for i in range(len(traj)-1):
      board_diff = traj[i][0].argmax(dim=0) - traj[i+1][0].argmax(dim=0)
      #print(traj[i][0].argmax(dim=0))
      #print(traj[i+1][0].argmax(dim=0))
      #print(board_diff)
      move = board_diff.nonzero()
      move = move[0,1].item()
      best_move = traj[i][1][:-1].argmax()
      if (move != best_move):
        subopt_counter += 1
        diff_counter += traj[i][1][:-1].max() - traj[i][1][:-1][move]
  print(f"subopt: {subopt_counter / move_counter}")
  print(f"diff: {diff_counter / subopt_counter}")
  print(f"Avg moves: {move_counter / len(trajs)}")

#analyze(begin_data)
#analyze(end_data)

#i = 1 / 0

#for i in range(20):
#  game = data[random.randint(0, len(data) - 1)]
#  #elt = game[random.randint(0, len(game) - 1)]
#  #elt = data[random.randint(0, len(data) - 1)][-1]
#  for elt in game:
#    viz(elt)

#print(f"Avg len: {sum([len(traj) for traj in ldata]) / len(ldata)}")
#data = pkl.load(open("a0_game_trajectories.pkl", "rb"))
#data = [[(torch.tensor(elt[0], dtype=torch.float32), 
#          torch.tensor(elt[1], dtype=torch.float32)) for elt in traj[-3:]] for traj in data]


  #return [[(torch.tensor(elt[0], dtype=torch.float32), 
  #            torch.tensor(elt[1] * (coeff ** i), dtype=torch.float32)) for i, elt in 
  #              enumerate(reversed(random.choices(traj,k=min(len(traj), samples))))] for traj in data]

#layer_size = [120] # [50, 90, 120, 175]
#for i in range(5):
#  for samples in [5]: #[1, 5, 15, 43]:
#    #for dataname, data in [("short", short_data)]: #, ("long", long_data)]:
#    for early in [True]:
def ident(state):
  return state

def trim(state):
  return state[[0, 2], :, :]

def inv(state):
  return (state.argmax(dim=0) - 1).float().unsqueeze(0)

print(len(data))

for i in range(1):
  for f in [0.5]:
  #for start_ind, end_ind, name in [(0, 4500, "pretrained_mixed"),
  #                         (1500, 4500, "pretrained_unmixed"),
  #                         (3000, 4500, "pretrained_unmixed_short"),
  #                         (0, 1500, "pretrained_retrain")]:
  #for width in [80]:
    #for drop in [0.4]:
    for batch_size in [100]:
      for v in [0.5]:
        for patience in [0]:
          #f = 0.5
          #data = short_data[start_ind:end_ind]
          #seed_everything(i + 21)
  #for transform_name, transform, channels in [
  #                                  ("negative", inv, 1),
  #                                  ("no-zero", trim, 2),
  #                                  ("one-hot", ident, 3),
  #                                  ]:
          #data = short_data[:500]
          #data = prep(data, 43)
          #data = [[(transform(elt[0]), elt[1]) for elt in traj] for traj in data]
          train_data, val_data = create_datasets(data, 
                                                    samples_per_game=None,
                                                    flip_prob=f, 
                                                    validation_games=0.1)
          extra_train_data = AlphaZeroDataset(short_data[:1500], None, f)
          train_data.data.extend(extra_train_data.data)

          #for i in range(20):
          #  game = data[random.randint(0, len(data) - 1)]
          #  elt = game[random.randint(0, len(game) - 1)]
          #  #elt = data[random.randint(0, len(data) - 1)][-1]
          #  viz(elt)
          #islong = dataname == "long"
          train_args = {
                    "log_dir": "alphazero_logs_v7",
                    "epochs_per_round": 15,
                    "batch_size": batch_size,
                    "value_weight": v,
                    "lr": 1e-3,
                    "l2_reg": 1e-3,
                    } 
          
          train_loader = DataLoader(train_data, shuffle=True, batch_size=train_args["batch_size"])
          val_loader = DataLoader(val_data, shuffle=False, batch_size=train_args["batch_size"])
          #name= "resnet-" + str(drop)
          #name = "early_stop_loss" #"early_stop" if early else "late_stop"
          early = patience != 0
          #name= "early_" + str(patience) if early else "late"
          name = "pretrained_mixed_valid" #"flip_" + str(f)
          for j in range(1):
            #target_model = AlphaZeroCNN(train_args["lr"], train_args["l2_reg"], train_args["value_weight"])
            target_model = AlphaZeroCNN.load_from_checkpoint("checkpoints_v7/0.ckpt")
            tb_logger = TensorBoardLogger(save_dir=train_args["log_dir"], name=name, default_hp_metric=False)
            early_stop_callback = EarlyStopping(monitor="val/value_loss", mode="min", patience=patience)
            cbs = [early_stop_callback] if early else []
            trainer = Trainer(max_epochs=train_args["epochs_per_round"],
                                  #min_epochs = 5,
                                  enable_checkpointing=False,
                                  accelerator="gpu", 
                                  devices=1,
                                  logger=tb_logger,
                                  val_check_interval=0.1,
                                  enable_model_summary = False,
                                  callbacks=cbs,
                                  #overfit_batches=0.01,
                                  #log_every_n_steps=20,
                                  )
            #trainer.tune(target_model, train_loader, val_loader)
            trainer.fit(target_model, train_loader, val_loader)
           # trainer.save_checkpoint("resnet_undisc.ckpt")




