import torch
import random
import copy
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.players.components.AlphaZeroNets import AlphaZeroFCN
from src.players.MCTS import MCTS
from src.ConnectFour import ConnectFour
from src.players.components.AlphaZeroDataset import AlphaZeroDataset
from src.players.components.checkpoint_utils import *
from src.players.AlphaZeroPlayer import eval_func

def one_hot_state(game):
  return np.eye(3)[game.board + 1]

def init_players(target_model, opponent_model, game_args):
  target_eval_func = lambda game, actions: eval_func(target_model, game, actions)
  opponent_eval_func = lambda game, actions: eval_func(opponent_model, game, actions)
  mcts_init_kwargs = game_args.copy()
  target_player = MCTS(target_eval_func, **mcts_init_kwargs)
  opponent_player = MCTS(opponent_eval_func, **mcts_init_kwargs)
  return target_player, opponent_player

def play_game(target_model, opponent_model, game_args, save_trajectory=True):
  state_trajectory = []
  action_score_trajectory = []
  is_over = False
  current_player, next_player = init_players(target_model, opponent_model, game_args)
  target_model_first = bool(random.getrandbits(1))
  if (not target_model_first):
    current_player, next_player = next_player, current_player
  game = ConnectFour()
  while (not is_over):
    action = current_player.search(game, game_args["mcts_iters"])
    action_scores = mcts.current_action_scores(game)
    is_target_move = game.player == (1 if target_model_first else -1)
    if (save_trajectory and is_target_move):
      state_trajectory.append(one_hot_state(game))
      action_score_trajectory.append(scores)
    game.perform_move(action)
    is_over, outcome = game.game_over()
    current_player, next_player = next_player, current_player
  outcome = outcome * (1 if target_model_first else -1)
  return outcome, state_trajectory, action_score_trajectory

def eval_new_model(target_model, opponent_model, game_args, num_eval_games, win_threshold):
  win_counter = 0
  for i in tqdm(range(num_eval_games), desc="eval games"):
    outcome, _, _ = play_game(target_model, opponent_model, game_args, save_trajectory=False)
    if (outcome == 1):
      win_counter += 1
  return float(win_counter) / float(num_eval_games) >= win_threshold

def train_model(model, data, outcomes, train_args):
  train_loader = DataLoader(data, shuffle=True, batch_size=train_args["batch_size"])
  trainer = pl.Trainer(max_epochs=train_args["epochs_per_round"], enable_checkpointing=False, accelerator="gpu", devices=1) # TODO set lr finding, epochs, hparams
  trainer.fit(train_loader)
  return trainer

def train(args):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  pl.seed_everything(args["seed"])
  latest_model = get_latest_model(args["checkpoint_dir"])
  for round_ind in tqdm(range(args["rounds"]), desc="rounds"):
    target_model = copy.deepcopy(latest_model).to(device)
    opponent_model = get_random_opponent(args["checkpoint_dir"]).to(device)
    game_trajectories = []
    for game_ind in tqdm(range(args["games_per_round"]), desc="training games"):
      outcome, state_trajectory, action_score_trajectory = \
        play_game(latest_model, opponent_model, args["game_cfg"])
      game_trajectories.append((outcome, state_trajectory, action_score_trajectory))
    data = AlphaZeroDataset(game_trajectories)
    trainer = train_model(target_model, data, outcomes, args["train_cfg"])
    if (eval_new_model(target_model, latest_model, args["game_cfg"], args["num_eval_games"], args["win_threshold"])):
      save_model(args["checkpoint_dir"], trainer)
      latest_model = target_model