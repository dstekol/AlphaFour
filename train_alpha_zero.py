import torch
import random
import copy
from tqdm import tqdm
from pytorch_lightning import seed_everything
from src.players.AlphaZeroPlayer import AlphaZeroPlayer, one_hot_state
from src.ConnectFour import ConnectFour
from src.players.components.checkpoint_utils import *
from src.players.components.ResignCounter import ResignCounter
from parse_utils import parse_args_trainer
from src.players.components.AlphaZeroDataset import AlphaZeroDataset
from src.players.components.EvalBuffer import EvalBuffer
from src.players.components.net_utils import train_model, init_model
import functools
import pickle as pkl

tqdm_slider = functools.partial(tqdm, leave=True, position=0)

def init_players(buffer_a, buffer_b, game_args):
  mcts_arg_names = ["explore_coeff", "mcts_iters", "temperature", "dirichlet_coeff", "dirichlet_alpha", "discount", "num_threads"]
  mcts_args = {arg_name: game_args[arg_name] for arg_name in mcts_arg_names}
  player_a = AlphaZeroPlayer(buffer_a, mcts_args)
  player_b = AlphaZeroPlayer(buffer_b, mcts_args)
  return player_a, player_b

def play_game(buffer_a, buffer_b, game_args, save_trajectory, resign_counter):
  # initialize vars
  trajectory = []
  is_over = False
  resignations = {1: False, -1: False}
  resign_allowed = random.random() > game_args["resign_forbid_prob"]
  game = ConnectFour()
  player_a, player_b = init_players(buffer_a, buffer_b, game_args)
  
  # randomly decide first player
  player_a_first = bool(random.getrandbits(1))
  if (player_a_first):
    current_player, next_player = player_a, player_b
  else:
    current_player, next_player = player_b, player_a
  
  while (not is_over):
    # set exploration temperature to 0 for later portion of game
    if (game.num_moves == game_args["temp_drop_step"]):
      current_player.drop_temperature()
      next_player.drop_temperature()
    action = current_player.pick_move(game)

    # handle resignations
    if (not resignations[game.player]
        and current_player.should_resign(game, game_args["resign_threshold"])):
      resignations[game.player] = True
      if (resign_allowed):
        outcome = -game.player
        break
    
    # save game trajectory for if needed
    if (save_trajectory):
      state = torch.tensor(one_hot_state(game), dtype=torch.float32)
      action_scores = current_player.mcts.current_action_scores(game, game_args["temperature"])
      target_output = torch.tensor(np.concatenate((action_scores, [0]), axis=0), dtype=torch.float32)
      trajectory.append((state, target_output))
    
    # game proceeds
    game.perform_move(action)
    is_over, outcome = game.game_over()
    current_player, next_player = next_player, current_player

  # update saved trajectories with outcome
  if (save_trajectory):
    for i, (state, target_output) in enumerate(trajectory):
      player = 1 if i % 2 == 0 else -1
      moves_until_end = len(trajectory) - i - 1
      target_output[-1] = outcome * player * (game_args["discount"] ** moves_until_end)
  
  # compute whether player A won (since order may have been reversed)
  player_a_won = outcome * (1 if player_a_first else -1)

  # check for false resignations
  if (resign_counter is not None):
    resign_counter.update(outcome, resignations, resign_allowed)

  return player_a_won, trajectory

def create_datasets(game_trajectories, samples_per_game, flip_prob, validation_games):
  if (validation_games is None):
    validation_games = 0
  elif (validation_games < 1):
    validation_games = int(len(game_trajectories) * validation_games)
  else: 
    validation_games = min(len(game_trajectories), validation_games)
  data_inds = np.random.permutation(len(game_trajectories))
  val_inds = data_inds[:validation_games]
  train_inds = data_inds[validation_games:]
  train_trajectories = [game_trajectories[i] for i in train_inds] 
  val_trajectories = [game_trajectories[i] for i in val_inds] 
  train_data = AlphaZeroDataset(train_trajectories, samples_per_game, flip_prob)
  val_data = AlphaZeroDataset(val_trajectories, samples_per_game, flip_prob)
  return train_data, val_data


def eval_new_model(target_buffer, opponent_buffer, game_args, num_eval_games, win_threshold):
  target_buffer.model.eval()
  opponent_buffer.model.eval()
  win_counter = 0
  game_args = game_args.copy()
  game_args["temp_drop_step"] = 0
  game_args["dirichlet_coeff"] = 0
  for i in tqdm_slider(range(num_eval_games), desc="eval games"):
    outcome, _ = play_game(target_buffer, 
                           opponent_buffer,
                           game_args, 
                           save_trajectory=False, 
                           resign_counter=None)
    if (outcome == 1):
      win_counter += 1

  frac_wins = float(win_counter) / float(num_eval_games)
  print(f"\nTarget model won {frac_wins} of eval games")
  keep_model = frac_wins >= win_threshold
  if (keep_model):
    print("\nAccepting new model")
  else:
    print("\nRejecting new model")
  return keep_model

def train(args):
  device = "cuda" if torch.cuda.is_available() and args["cuda"] else "cpu"
  seed_everything(args["seed"])

  # retrieve current best model (or initialize if no previous models)
  best_model = get_latest_model(args["checkpoint_dir"])
  if (best_model is None):
    print("\nPitting against newly initialized model")
    best_model = init_model(args["train_args"], device) # AlphaZeroCNN(**args["train_args"])

  game_trajectories = load_game_trajectories(args["games_input_file"])

  for round_ind in tqdm_slider(range(args["round_start_ind"], args["round_start_ind"] + args["rounds"]), desc="rounds"):
    print(f"\nRound: {round_ind}")
    resign_counter = ResignCounter()

    # perform self-play games and collect play data
    best_model.eval()
    model_buffer = EvalBuffer(best_model, args["max_buffer_size"], args["max_wait_time"]) #TODO
    for game_ind in tqdm_slider(range(args["games_per_round"]), desc="training games"):
      outcome, trajectory =  play_game(model_buffer, 
                                       model_buffer, 
                                       args["game_args"], 
                                       save_trajectory=True, 
                                       resign_counter=resign_counter)
      game_trajectories.append(trajectory)
      if (len(game_trajectories) > args["max_queue_len"]):
        game_trajectories.pop(0)
    model_buffer.close()

    if (args["games_output_file"] is not None):
        pkl.dump(game_trajectories, open(args["games_output_file"], "wb"))
    resign_counter.print_stats()

    # train policy and value networks to predict action and state scores respectively
    train_data, val_data = create_datasets(game_trajectories, 
                                           args["samples_per_game"], 
                                           args["flip_prob"], 
                                           args["validation_games"])
    print(f"Len train data: {len(train_data)}")
    target_model, save_checkpoint_handle = train_model(best_model, 
                                         train_data, 
                                         val_data, 
                                         round_ind, 
                                         args["train_args"], 
                                         device)

    # evaluate newly trained model against previous best model
    target_buffer = EvalBuffer(target_model, args["max_buffer_size"], args["max_wait_time"])
    opponent_buffer = EvalBuffer(best_model, args["max_buffer_size"], args["max_wait_time"])
    keep_new_model = eval_new_model(target_buffer,
                       opponent_buffer, 
                       args["game_args"], 
                       args["eval_games"], 
                       args["win_threshold"])
    target_buffer.close()
    opponent_buffer.close()
    if (keep_new_model):
      save_model(args["checkpoint_dir"], save_checkpoint_handle)
      best_model = target_model

if __name__ == "__main__":
  args = parse_args_trainer()
  train(args)