import torch
import random
import copy
from tqdm import tqdm
from pytorch_lightning import seed_everything
from src.players.AlphaZeroPlayer import AlphaZeroPlayer, one_hot_state
from src.ConnectFour import ConnectFour
from src.players.alphazero.checkpoint_utils import *
from src.players.alphazero.ResignCounter import ResignCounter
from parse_utils import parse_args_trainer, copy_args
from src.players.alphazero.BufferedModelWrapper import BufferedModelWrapper
from src.players.alphazero.net_utils import train_model_multi_version, init_model
from src.players.alphazero.Trajectory import Trajectory
from validation_utils import AgentType
import functools
import pickle as pkl
import warnings

tqdm_slider = functools.partial(tqdm, leave=True, position=0)


def init_players(buffered_model_a, buffered_model_b, game_args):
  """
  Initializes two AlphaZero player objects with GPU buffers corresponding to buffer_a and buffer_b,
  and gameplay parameters initialized according to values in game_args

  Args:
  buffered_model_a (EvalBuffer): buffer object corresponding to player a (for queueing game states for evaluation by neural network, used in place of actual network)
  buffered_model_b (EvalBuffer): buffer object corresponding to player b (for queueing game states for evaluation by neural network, used in place of actual network)
    (can be same as buffer_a if using same underlying model)
  game_args (Dict[String, Any]): dictionary containing subset of command line arguments, with the following items:
    explore_coeff, mcts_iters, temperature, dirichlet_coeff, dirichlet_alpha, discount, num_threads

  Returns:
  player_a (AlphaZeroPlayer) - AlphaZeroPlayer object corresponding to player a
  player_b (AlphaZeroPlayer) - AlphaZeroPlayer object corresponding to player b
  """

  mcts_arg_names = ["explore_coeff", "mcts_iters", "temperature", "dirichlet_coeff", "dirichlet_alpha", "discount", "num_threads"]
  mcts_args = copy_args(game_args, mcts_arg_names)
  player_a = AlphaZeroPlayer(buffered_model_a, mcts_args)
  player_b = AlphaZeroPlayer(buffered_model_b, mcts_args)
  return player_a, player_b

def play_game(buffered_model_a, buffered_model_b, game_args, save_trajectory, resign_counter):
  """
  Plays a game between player A (corresponding to buffer_a) and player B (corresponding to buffer_b), 
  and returns whether player A won (and optionally the game trajectory). 
  Initializes temperature to game_args["temperature"], then drops the temperature to game_args["drop_temperature"] at timestep game_args["temp_drop_step"]

  Args:
  buffered_model_a (BufferedModelWrapper): buffer object corresponding to player a (for queueing game states for evaluation by neural network, used in place of actual network)
  buffered_model_b (BufferedModelWrapper): buffer object corresponding to player b (for queueing game states for evaluation by neural network, used in place of actual network)
    (can be same as buffer_a if using same underlying model)
  game_args (Dict[String, Any]): dictionary containing subset of command line arguments, with the following items:
    explore_coeff, mcts_iters, temperature, dirichlet_coeff, dirichlet_alpha, discount, num_threads
  save_trajectory (Boolean): whether or not to return the game trajectory (saved as a list of state-outcome tuples)
  resign_counter (ResignCounter): ResignCounter object for tracking resignation statistics

  Returns:
  player_a_won (Boolean): Whether or not the player corresponding to buffer_a won the game. 
    1 if player A won, -1 if player B won, 0 if tie.
  trajectory (List[Tuple[torch.tensor, torch.tensor]]): List of state-outcome tuples representing the game trajectory.
    First element in each tuple is one-hot representation of game state. 
    Second element is the model output target: an 8x1 tensor where the first 7 elements correspond to MCTS action scores, 
    and the 8th element is the game outcome (discounted according to value of game_args["discount"]).
    If save_trajectory parameter is False, empty list will be returned.
  """
  # initialize vars
  trajectory = Trajectory()
  is_over = False
  resignations = {1: False, -1: False}
  resign_allowed = random.random() > game_args["resign_forbid_prob"]
  game = ConnectFour()
  player_a, player_b = init_players(buffered_model_a, buffered_model_b, game_args)
  
  # randomly decide first player
  player_a_first = bool(random.getrandbits(1))
  if (player_a_first):
    current_player, next_player = player_a, player_b
  else:
    current_player, next_player = player_b, player_a
  
  while (not is_over):
    # set exploration temperature to 0 for later portion of game
    if (game.num_moves == game_args["temp_drop_step"]):
      current_player.update_temperature(game_args["drop_temperature"])
      next_player.update_temperature(game_args["drop_temperature"])
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=UserWarning)
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




def eval_new_model(target_buffer, opponent_buffer, game_args, num_eval_games):
  """
  Plays a number of evaluation games (dictated by num_eval_games) pitting the target model 
  (corresponding to target_buffer) against the opponent/baseline buffer (corresponding to opponent_buffer)
  and returns the win percentage of the target model. 
  Temperature is set to 0 (max-score move is picked deterministically) and dirichlet noise is deactivated 
  (dirichlet_coeff is set to 0) regardless of values in game_args dictionary.

  Args:
  target_buffer (EvalBuffer): buffer object corresponding to the target model to be evaluated (for queueing game states for evaluation by neural network, used in place of actual network)
  opponent_buffer (EvalBuffer): buffer object corresponding to the opponent/baseline model, which is the previous best model (for queueing game states for evaluation by neural network, used in place of actual network)
    (can be same as buffer_a if using same underlying model)
  game_args (Dict[String, Any]): dictionary containing subset of command line arguments, with the following items:
    explore_coeff, mcts_iters, temperature, dirichlet_coeff, dirichlet_alpha, discount, num_threads
  num_eval_games: The number of evaluation games to play

  Returns:
  frac_wins (Float): Number in range [0,1] representing the fraction of the evaluation games that the target model won
  """
  target_buffer.model.eval()
  opponent_buffer.model.eval()
  win_counter = 0
  game_args = game_args.copy()
  game_args["temperature"] = 0
  game_args["temp_drop_step"] = None
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
  return frac_wins

def train(args):
  init_dirs(args["base_dir"])
  save_args(args)
  device = "cuda" if torch.cuda.is_available() and args["cuda"] else "cpu"
  seed_everything(args["seed"])

  # retrieve current best model (or initialize if no previous models)
  best_model, name = get_latest_model(args["base_dir"], device)
  if (best_model is None):
    print("No saved models")
    best_model = init_model(args["net_args"], device)
  else:
    print(f"Pitting against version: {name}")

  trajectories, init_round_num = load_game_trajectories(args["base_dir"])
  if (len(trajectories) == 0):
    print("No saved trajectories")
  else:
    print(f"Loading saved trajectories: {init_round_num}")  

  for round_ind in tqdm_slider(range(init_round_num + 1, init_round_num + 1 + args["rounds"]), desc="rounds"):
    print(f"\nRound: {round_ind}")
    print(f"Device: {best_model.device}")
    resign_counter = ResignCounter()

    # perform self-play games and collect play data
    best_model.eval()
    
    model_buffer = BufferedModelWrapper(best_model, args["max_buffer_size"], args["max_wait_time"])
    for game_ind in tqdm_slider(range(args["games_per_round"]), desc="training games"):
      outcome, trajectory =  play_game(model_buffer, 
                                        model_buffer, 
                                        args["game_args"], 
                                        save_trajectory=True, 
                                        resign_counter=resign_counter)
      trajectories.append(trajectory)
      if (len(trajectories) > args["max_queue_len"]):
        trajectories.pop(0)
    model_buffer.close()

    #if (args["games_output_file"] is not None)
    save_game_trajectories(args["base_dir"], trajectories, round_ind) 
      #pkl.dump(trajectories, open(args["games_output_file"], "wb"))
    
    avg_moves = sum([len(traj) for traj in trajectories[-args["games_per_round"]:]]) / args["games_per_round"]
    print(f"Avg Moves: {avg_moves:.3f}")
    print(str(resign_counter.get_stats()))
    

    # train policy and value networks to predict action and state scores respectively
    #train_data, val_data = create_datasets(game_trajectories, 
    #                                       args["samples_per_game"], 
    #                                       args["flip_prob"], 
    #                                       args["validation_games"])
    dataset_arg_names = ["samples_per_game", "flip_prob", "validation_games"]
    dataset_args = copy_args(args, dataset_arg_names)
    target_model, save_checkpoint_handle = train_model_multi_version(best_model, 
                                         trajectories, 
                                         dataset_args,
                                         round_ind, 
                                         args["net_args"], 
                                         args["base_dir"],
                                         device)

    # evaluate newly trained model against previous best model
    target_buffer = BufferedModelWrapper(target_model, args["max_buffer_size"], args["max_wait_time"])
    opponent_buffer = BufferedModelWrapper(best_model, args["max_buffer_size"], args["max_wait_time"])
    frac_wins = eval_new_model(target_buffer,
                       opponent_buffer, 
                       args["game_args"], 
                       args["eval_games"])
    keep_new_model = frac_wins >= args["win_threshold"]
    target_buffer.close()
    opponent_buffer.close()
    if (keep_new_model):
      print("\nAccepting new model")
      save_model(args["base_dir"], save_checkpoint_handle, round_ind)
      best_model = target_model
    else:
      print("\nRejecting new model")
    
    stats = {"Frac wins": frac_wins, "Avg moves": avg_moves, "Accepted": keep_new_model}
    stats.update(resign_counter.get_stats())
    log_stats(args["base_dir"], round_ind, stats)

if __name__ == "__main__":
  args = parse_args_trainer()
  train(args)

