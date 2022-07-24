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

def init_players(target_model, opponent_model, game_args):
  mcts_arg_names = ["explore_coeff", "mcts_iters", "temperature", "dirichlet_coeff", "dirichlet_alpha"]
  mcts_args = {arg_name: game_args[arg_name] in mcts_arg_names}
  target_player = AlphaZeroPlayer(target_model, mcts_args)
  opponent_player = AlphaZeroPlayer(opponent_model, mcts_args)
  return target_player, opponent_player

def play_game(target_model, opponent_model, game_args, save_trajectory=True):
  # initialize vars
  state_trajectory = []
  action_score_trajectory = []
  is_over = False
  resignations = [False, False]
  game = ConnectFour()
  target_player, opponent_player = init_players(target_model, opponent_model, game_args)
  
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
  game_args = game_args.copy()
  game_args["temp_drop_step"] = 0
  game_args["dirichlet_coeff"] = 0
  for i in tqdm(range(num_eval_games), desc="eval games", leave=True, position=0):
    outcome, _, _, _ = play_game(target_model, opponent_model, game_args, save_trajectory=False)
    if (outcome == 1):
      win_counter += 1
  return float(win_counter) / float(num_eval_games) >= win_threshold

def train_model(model, train_data, val_data, train_args, device):
  train_loader = DataLoader(train_data, shuffle=True, batch_size=train_args["batch_size"])
  val_loader = DataLoader(val_data, shuffle=False, batch_size=train_args["batch_size"])
  trainer = pl.Trainer(max_epochs=train_args["epochs_per_round"], 
                       enable_checkpointing=False,
                       accelerator=("gpu" if device=="cuda" else "cpu"), 
                       devices=1,
                       limit_val_batches=0,
                       num_sanity_val_steps=0)
  trainer.fit(model, train_loader, val_loader)
  return trainer

def preprocess_game_data(game_trajectories, samples_per_game, flip_prob, validation_games):
  data_inds = np.random.permutation(len(game_trajectories))
  if (validation_games is None):
    validation_games = 0
  elif (validation_games < 1):
    validation_games = int(len(game_trajectories) * validation_games)
  else: 
    validation_games = min(len(game_trajectories), validation_games)
  train_trajectories = game_trajectories[data_inds[:validation_games]]
  val_trajectories = game_trajectories[data_inds[validation_games:]]
  train_data = AlphaZeroDataset(train_trajectories, args["samples_per_game"], args["flip_prob"])
  val_data = AlphaZeroDataset(val_trajectories, args["samples_per_game"], args["flip_prob"])
  return train_data, val_data


def train(args):
  device = "cuda" if torch.cuda.is_available() and args["cuda"] else "cpu"
  pl.seed_everything(args["seed"])

  # retrieve current best model (or initialize if no previous models)
  latest_checkpoint_model = get_latest_model(args["checkpoint_dir"])
  if (latest_checkpoint_model is None):
    latest_checkpoint_model = AlphaZeroFCN(**args["train_args"])

  for round_ind in tqdm(range(args["rounds"]), desc="rounds", leave=True, position=0):
    wrong_resigns = 0
    total_resigns = 0

    # initialize training model to current best model
    target_model = copy.deepcopy(latest_checkpoint_model).to(device)

    # choose random opponent from set of previous models (or initialize if no previous models)
    opponent_model, opponent_name = get_random_opponent(args["checkpoint_dir"])
    if (opponent_model is None):
      opponent_model = AlphaZeroFCN(**args["train_args"])
      print("\nPitting against fresh model")
    else:
      print("\nPitting against: " + opponent_name)
    opponent_model = opponent_model.to(device)

    # perform self-play games and collect play data
    game_trajectories = []
    for game_ind in tqdm(range(args["games_per_round"]), desc="training games", leave=True, position=0):
      outcome, state_trajectory, action_score_trajectory, resignations = \
        play_game(latest_checkpoint_model, opponent_model, args["game_args"])
      game_trajectories.append((outcome, state_trajectory, action_score_trajectory))

      # update resignation stats
      if (resignations[0]):
        total_resigns += 1
        if (outcome != -1):
          wrong_resigns += 1
    
    # print resignation stats
    resign_frac = float(total_resigns) / float(args["games_per_round"])
    print(f"\nResignation proportion: {resign_frac}")
    if (wrong_resigns > 0):
      wrong_resign_frac = float(wrong_resigns) / float(total_resigns)
      print(f"\nFalse resignation proportion: {wrong_resign_frac}")

    # train policy and value networks to predict action and state scores respectively
    train_data, val_data = preprocess_game_data(game_trajectories, args["samples_per_game"], args["flip_prob"], args["validation_games"])
    trainer = train_model(target_model, train_data, val_data, args["train_args"], device)

    # evaluate newly trained model against previous best model
    keep_new_model = eval_new_model(target_model, 
                       latest_checkpoint_model, 
                       args["game_args"], 
                       args["eval_games"], 
                       args["win_threshold"])
    if (keep_new_model):
      print("\nCheckpointing new model") 
      save_model(args["checkpoint_dir"], trainer)
      latest_checkpoint_model = target_model
    else:
      print("\nRejecting new model")

if __name__ == "__main__":
  args = parse_args_trainer()
  train(args)