import torch
import random
import copy
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.players.components.AlphaZeroNets import AlphaZeroFCN
from src.players.AlphaZeroPlayer import AlphaZeroPlayer, one_hot_state
from src.ConnectFour import ConnectFour
from src.players.components.AlphaZeroDataset import AlphaZeroDataset
from src.players.components.checkpoint_utils import *
from parse_utils import parse_args_trainer

def play_game(target_model, opponent_model, game_args, save_trajectory=True):
  # initialize vars
  state_trajectory = []
  action_score_trajectory = []
  is_over = False
  resignations = [False, False]
  game = ConnectFour()
  target_player = AlphaZeroPlayer(target_model, 
                                  explore_coeff=game_args["explore_coeff"], 
                                  mcts_iters=game_args["mcts_iters"], 
                                  temperature=game_args["temperature"])
  opponent_player = AlphaZeroPlayer(opponent_model, 
                                    explore_coeff=game_args["explore_coeff"], 
                                    mcts_iters=game_args["mcts_iters"], 
                                    temperature=game_args["temperature"])
  
  # randomly decide first player
  target_player_first = bool(random.getrandbits(1))
  if (target_player_first):
    current_player, next_player = target_player, opponent_player
  else:
    current_player, next_player = opponent_player, target_player
  
  while (not is_over):
    # set exploration temperature to 0 for later portion of game
    if (game.num_moves == game_args["temp_drop_step"]):
      current_player.drop_temperature()
      next_player.drop_temperature()
    action = current_player.pick_move(game)

    # handle resignations
    if (current_player.should_resign(game, game_args["resign_threshold"])):
      resignations[0 if current_player ==  target_player else 1] = True
      if (random.random() > game_args["resign_forbid_prob"]):
        outcome = -game.player
        break
    
    # save game trajectory for target player if requested
    if (save_trajectory and current_player == target_player):
      state_trajectory.append(one_hot_state(game))
      action_scores = current_player.mcts.current_action_scores(game)
      action_score_trajectory.append(action_scores)
    
    # game proceeds
    game.perform_move(action)
    is_over, outcome = game.game_over()
    current_player, next_player = next_player, current_player
  
  # check whether target player won (depending on who went first)
  outcome = outcome * (1 if target_player_first else -1)
  return outcome, state_trajectory, action_score_trajectory, resignations

def eval_new_model(target_model, opponent_model, game_args, num_eval_games, win_threshold):
  win_counter = 0
  for i in tqdm(range(num_eval_games), desc="eval games"):
    outcome, _, _, _ = play_game(target_model, opponent_model, game_args, save_trajectory=False)
    if (outcome == 1):
      win_counter += 1
  return float(win_counter) / float(num_eval_games) >= win_threshold

def train_model(model, data, train_args, device):
  train_loader = DataLoader(data, shuffle=True, batch_size=train_args["batch_size"])
  trainer = pl.Trainer(max_epochs=train_args["epochs_per_round"], 
                       enable_checkpointing=False,
                       accelerator=("gpu" if device=="cuda" else "cpu"), 
                       devices=1,
                       limit_val_batches=0,
                       num_sanity_val_steps=0)
  trainer.fit(model, train_loader)
  return trainer

def train(args):
  device = "cuda" if torch.cuda.is_available() and args["cuda"] else "cpu"
  pl.seed_everything(args["seed"])

  # retrieve current best model (or initialize if no previous models)
  latest_checkpoint_model = get_latest_model(args["checkpoint_dir"])
  if (latest_checkpoint_model is None):
    latest_checkpoint_model = AlphaZeroFCN(**args["train_args"])

  for round_ind in tqdm(range(args["rounds"]), desc="rounds"):
    wrong_resigns = 0
    total_resigns = 0

    # initialize training model to current best model
    target_model = copy.deepcopy(latest_checkpoint_model).to(device)

    # choose random opponent from set of previous models (or initialize if no previous models)
    opponent_model, opponent_name = get_random_opponent(args["checkpoint_dir"])
    if (opponent_model is None):
      opponent_model = AlphaZeroFCN(**args["train_args"])
      print("Pitting against fresh model")
    else:
      print("Pitting against: " + opponent_name)
    opponent_model = opponent_model.to(device)

    # perform self-play games and collect play data
    game_trajectories = []
    for game_ind in tqdm(range(args["games_per_round"]), desc="training games"):
      outcome, state_trajectory, action_score_trajectory, resignations = \
        play_game(latest_checkpoint_model, opponent_model, args["game_args"])
      game_trajectories.append((outcome, state_trajectory, action_score_trajectory))

      # update resignation stats
      if (resignations[0]):
        total_resigns += 1
        if (outcome != -1):
          wrong_resigns += 1
    
    # print fraction of false positive resignations
    if (wrong_resigns > 0):
      wrong_resign_frac = float(wrong_resigns) / float(total_resigns)
      print(f"False resignation proportion: {wrong_resign_frac}")

    # train policy and value networks to predict action and state scores respectively
    data = AlphaZeroDataset(game_trajectories, args["samples_per_game"], args["flip_prob"])
    trainer = train_model(target_model, data, args["train_args"], device)

    # evaluate newly trained model against previous best model
    keep_new_model = eval_new_model(target_model, 
                       latest_checkpoint_model, 
                       args["game_args"], 
                       args["eval_games"], 
                       args["win_threshold"])
    if (keep_new_model):
      print("Checkpointing new model")
      save_model(args["checkpoint_dir"], trainer)
      latest_checkpoint_model = target_model
    else:
      print("Rejecting new model")

if __name__ == "__main__":
  args = parse_args_trainer()
  train(args)