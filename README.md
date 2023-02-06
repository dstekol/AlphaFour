# Connect Four Agents (AlphaFour)
This repository contains a variety of agents designed to play Connect Four. The available agents, 
in order of implementation complexity, are:
- **RandomAgent**: fairly self-explanatory...
- **AlphaBetaAgent**: performs depth-limited tree search with alpha-beta pruning and a handcrafted evaluation function
- **MCTSAgent**: performs full-depth Monte-Carlo simulations
- (most importantly) **AlphaZeroAgent**: based on the **AlphaZero** reinforcement learning algorithm from DeepMind.

The AlphaZero implementation includes nearly all of the features described in the paper, 
such as virtual loss, "multithreaded" (in as much as Python allows) MCTS simulations with GPU buffering to overcome bottlenecks, in-episode exploration temperature variation, dirichlet noise, 
false resignation monitoring, L2 network regularization, symmetry-based data augmentation, etc. 
Some minor departures have been made due to convenience and resource limitations, 
with the main caveat being that the agent is not designed to be trained across multiple machines.

The project exists mainly for the sake of exploring and comparing various gameplay strategies in a context 
which is reasonably, but not overwhelmingly, complex - however, there is also something to be said for the 
pleasure of playing against an algorithm that you actually understand. 
The goal is **not** to produce the most perfect/unbeatable Connect Four algorithm possible, as this has already been achieved elsewhere by other (less interesting) techniques.
One could argue that this code sits squarely at the intersection of research and revelry.

## Game Implementation
This repo contains a simple, no-frills implementation of the Connect Four game.
The current board state can be printed to the command line, and the game object exposes an interface for 
performing moves and checking whether the game has ended.
In particular, the game_over check has been highly optimized, in as much as Python allows, since it is performed at every move 
(and, for some of the agents, many thousands of times per user-facing move), which means it has the potential to be a major bottleneck.
If anyone else out there has spent as much time as I have profiling the performance of various Connect Four win-checking algorithms, you have my sympathy and my support.

## Agents

### RandomAgent
This agent places pieces at random, following either a uniform policy 
(pieces are placed in any open column with equal probability) or a gaussian policy 
(moves are biased toward the center, which is slightly more favorable than a uniform weighting).
It is both as fast and as clueless as you would expect.

### AlphaBetaAgent
This agent uses depth-limited Minimax search with alpha-beta pruning 
(meaning that low-reward actions are discarded if a different action yielding a higher reward has already been found). 
When the maximum search depth is reached, 
the board position is evaluated according to a handcrafted evaluation function which accounts for the 
position of "threats" (arrangements with three collinear same-color pieces, where the fourth piece cannot yet be placed due to an open column), 
and which uses a discount factor to encourage reaching favorable positions as quickly as possible.

The theory behind the handcrafted evaluation function is as follows: 
apart from tactics which are relatively simple to assess by looking ahead a few moves (like setting up a double-trap),
advanced Connect Four gameplay relies heavily on "threats", which are essentially traps for the other player - 
the trap-setter arranges their pieces in such a way that, if their opponent drops a piece in a specific column, 
they thereby fill in the gap below a cell which is needed to complete the trap-setter's connect-four, thus ending the game.
Setting up threats has long term consequences: 
a threat created at the beginning of the game can become the crucial factor toward the end of the game, once the boards has mostly been filled up and only the "danger" column is left open.
In particular, to maximize the chances that one's opponent will eventually be forced into the trap, the first player should strive to place a threat so that the open cell is on an odd row,
whereas the second player should strive for an even row. 
In theory, with perfect play, the first player can always achieve an odd-row threat and claim victory, 
but humans are rarely perfect, which keeps the game interesting.

### MCTSAgent
This agent selects moves by performing a large number of game "rollouts" (simulations), and progressively biasing itself toward moves that have yielded good results in the simulations. 
It follows the Monte Carlo Tree Search variant proposed in the AlphaZero paper, 
with the exception that, in the absence of an evaluation function, it uses a fixed prior distribution over moves 
(either uniform or gaussian, as with the RandomAgent), 
and performs full-depth simulations rather than using a value function as a short-circuit.
The agent then chooses the action corresponding to the highest visit count.
The MCTS search algorithm also forms the basis for the AlphaZero agent described below.
When using several thousand rollouts (ex. 4000), the MCTSAgent is, in the experience of the author, quite difficult to beat. 
I've done it a couple of times, but it hasn't been easy, and I'm not exactly a novice at Connect Four...

### AlphaZeroAgent
This agent is a from-scratch implementation (modulo PyTorch/PyTorch Lightning) of the AlphaZero reinforcement learning algorithm
(with the main caveat being that it is not designed to be trained or run across multiple computers). 
Multithreading and buffering is used to overcome GPU bottlenecks.


Each round of the training process consists of a fixed number of self-play games against a previous version of the agent, after which the agent's network is retrained so that 
its policy estimates better match the action scores previously computed by the MCTS module, and the 
value estimates better match the actual outcomes of the games. 
After each round, the newly trained agent is evaluated in a series of games against the previous best agent - 
if the new agent wins by a minimum percentage (ex 55%), then it is checkpointed and becomes the new best agent 
(and is used to evaluate subsequent agents).
During self-play agents are able to resign if the value of the current state and all of its immediate children 
is below a particular threshold - this measure is intended simply to save computation. 
During self-play, exploration is encouraged in several ways: 
 - The actions during MCTS simulations are chosen according a score which includes an exploration bonus. 
 This bonus decays each time an action is chosen, meaning less-frequently selected actions will eventually be selected.
 - At each self-play move (after the MCTS simulations are complete), actions are chosen probabilistically 
based on MCTS scores, which are computed based on visit counts but controlled by a "temperature" parameter - 
a low temperature biases the agent toward the highest-scoring action,
whereas a high temperature increases the probability of low-scoring actions being selected.
 - The MCTS scores are further perturbed via a small amount of Dirichlet noise.

A trained checkpoint (`alphazero_cnn.ckpt`) is included with the code. Due to computational resource limitations, the trained model does not achieve superhuman performance, but it is reasonably good against a casual player.


## Running the code
Users have the option of playing a two player game (with no AI algorithm on the other end), 
playing one-on-one against one of the agents, watching a game played between two agents, 
or training the AlphaZero agent from scratch.
For convenience, the repository includes a precomputed starter move tree for the MCTS agent 
(to speed up the computation for the first few moves), and a pre-trained neural network for the AlphaZero agent.

### Dependencies
 - [PyTorch](https://pytorch.org/)
 - [PyTorch Lightning](https://www.pytorchlightning.ai/)

### Two Player Game
To simply play a game against another human, users can run `run_two_player.py` script (no command line arguments are needed).
Each move is specified by typing in a number from 1 to 7, corresponding to the column.
Why anyone would want to use the command line interface for a two-player game instead of a physical board (or at least a GUI) 
is question mere mortals most likely cannot answer, but the functionality is there for those who want it.

### One Player Game
To play against a particular agent, users should run the `run_single_player.py` file with the `--agent` argument 
(set to one of "random", "alphabeta", "mcts", or "alphazero"), and any of the following agent-specific arguments:

#### RandomAgent arguments
- `--gaussian` (default = True) - Whether to use a Gaussian prior when choosing moves (thus biasing moves toward the center, which is generally slightly better). If set to False, a uniform distribution will be used instead.

#### AlphaBeta arguments
- `--quick-search-depth` (default=4) - the maximum tree depth when performing a quick-search (checking for obvious moves at the start of each turn)
- `--max-depth` (default=6) - the maximum tree depth to descend to before applying heuristic evaluation functions
- `--discount` (default=0.96) - the discount factor to apply to rewards (to encourage quick wins rather than dragging the game out)

#### MCTS arguments
- `--gaussian` (default = True) - Whether to use a Gaussian prior when choosing moves (thus biasing moves toward the center, which is generally slightly better). If set to False, a uniform distribution will be used instead.
- `--num-threads` (default=45) - Number of threads for MCTS
- `--mcts-iters` (default=250) - Number of MCTS/PUCT rollout simulations to execute for each move
- `--discount` (default=0.96) - Per-step discount factor for rewards (to encourage winning quickly)
- `--explore-coeff` (default=1) - Exploration coefficient for MCTS/PUCT search
- `--temperature` (default=0.8) - MCTS/PUCT exploration temperature setting (before temp-drop step)
- `--dirichlet-coeff` (default=0.02) - Dirichlet noise coefficient (added to action scores). If 0, no dirichlet noise will be added to MCTS scores; if 1, only dirichlet noise will be used.
- `--dirichlet-alpha` (default=0.3) - Dirichlet noise distribution parameter

#### AlphaZero arguments
- all MCTS arguments (except `--gaussian`)
- `--cuda` (default=True) - Whether to use CUDA acceleration
- `--checkpoint` (**Required**) - path to saved model checkpoint
- `--max-buffer-size` (default=20) - Maximum GPU buffer size (should be at most number of threads)
- `--max-wait-time` (default=0.05) - Maximum amount of time  (in milliseconds) to wait before flushing GPU buffer



### Pitting Agents Against One Another (Zero Player Game?)
To watch two agents play against each other, run `showdown.py` with the arguments `--agent1args` and `agent2args`, 
each of which should point to a file containing agent-specific command line arguments 
(at the least, each file must contain an `--agent` argument specifying the agent type as one of "random", "alphabeta", "mcts", "alphazero")


### Train AlphaZero
Run `train_alpha_zero.py` with the command line arguments below. The only required argument is `base_dir`, which specifies the base logging directory.
Within the base logging folder:
- game trajectories will be saved as pickle files to the `games` subfolder
- model checkpoints will be saved within the `models` subfolder as PyTorch Lightning archives with the .ckpt extension
- Tensorboard logs will be saved to the `logs` subfolder
- command line arguments and training statistics (ex. avg game length) will be saved to a `info.txt` file within the base folder

Command line arguments:
- `--base-dir` (**Required**) - Directory for model checkpoints (if checkpoints already exist in this directory, the trainer will use them as a starting point)
- `--seed` (default=42) - Random seed for reproducibility
- `--cuda` (default=True) - Whether to use CUDA acceleration
- `--rounds` (default=20) - Number of policy improvement rounds
- `--games-per-round` (default=2000) - Number of self-play games to execute per policy improvement round
- `--eval-games` (default=100) - Number of evaluation games to play between newly trained model and previous best model
- `--win-threshold` (default=0.55) - Fraction of evaluation games that newly trained model must win to replace previous best model
- `--flip-prob` (default=0.5) - Probability that input is flipped while training neural network (for data augmentation)
- `--samples-per-game` (default=None) - Number of state-outcome pairs per game trajectory to sample for training. If None, all data will be used. If in range (0,1), corresponding fraction of trajectories will be used. If integer greater than or equal to 1, corresponding number of games per trajectory will be used.
- `--validation-games` (default=0.05) - Number of self-play games to hold back for validation during neural network training.
- `--max-queue-len` (default=6000) - Maximum number of self-play games to retain in the training queue
- `--max-buffer-size` (default=20) - Maximum GPU buffer size (should be at most number of threads)
- `--max-wait-time` (default=0.05) - Maximum amount of time  (in milliseconds) to wait before flushing GPU buffer

- `--max-epochs` (default=15) - Max number of backpropagation epochs to train model on data collected from each round
- `--min-epochs` (default=5) - Min number of backpropagation epochs to train model on data collected from each round
- `--batch-size` (default=100) - Batch size for training neural network
- `--state-value-weight` (default=0.5) - Weight to put on value prediction relative to policy prediction (1 means all weight on value, 0 means all weight on policy)
- `--lr` (default=1e-3) - Learning rate for training neural network
- `--l2-reg` (default=1e-5) - Strength of L2 regularization for neural network
- `--patience` (default=None) - Number of non-improving steps to wait before stopping training
- `--train-attempts` (default=4) - Number of training runs to perform at each iteration (best model is selected based on validation loss)

- `--num-threads` (default=45) - Number of threads for MCTS
- `--mcts-iters` (default=250) - Number of PUCT simulations to execute for each move
- `--discount` (default=0.96) - Per-step discount factor for rewards (to encourage winning quickly)
- `--explore-coeff` (default=1) - Exploration coefficient for MCTS/PUCT search
- `--temperature` (default=0.8) - MCTS/PUCT exploration temperature setting (before temp-drop step)
- `--drop-temperature` (default=0.05) - MCTS/PUCT exploration temperature (after temp-drop step)
- `--dirichlet-coeff` (default=0.02) - Dirichlet noise coefficient (added to action scores)
- `--dirichlet-alpha` (default=0.3) - Dirichlet noise distribution parameter
- `--temp-drop-step` (default=7) - The episode step at which to drop the temperature (exploration strength) during self-play training games. This encourages exploration early in the game and stronger play later in the game.
- `--resign-threshold` (default=-0.85) - Threshold value below which agent will resign
- `--resign-forbid-prob` (default=0.1) - Probability that resignation will not be allowed (used in training to prevent false positives)


## Entertaining/Evil/Frustrating Mistakes Made & Bugs Encountered
 - The MCTS game outcome at the end of each simulation was originally negated, meaning the algorithm initially 
 strove to do everything possible to lose (and tended to achieve this goal). 
 On the bright-side, flipping the negative sign immediately resulted in much better play, 
 making it easily the single most low-effort/high-return fix of the project.

 - The neural network policy scores were being softmaxed during backpropagation training but not during gameplay,
 meaning the network was not outputting the distributions it was trained to during actual games. 
 Gameplay was bad, but not noticeably enough to immediately make the error obvious.

 - The state representation being passed into the network initially did not include whose turn it was, 
 meaning the network had to guess the outcome of the game without knowing which player was about to move.
 Though it is sometimes (but not always!) possible to deduce the current player based on the piece counts, 
 that is hardly a neural-net-friendly setup (learning to count is a whole 'nother problem), 
 and it predictably resulted in miserable performance.

 - A literal bug was encountered on the computer screen. Firm but cautious shoo-ing ensued.

 - In attempt to apply data augmentation, the board was being flipped upside-down instead of side to side.
 Connect Four is very much not invariant to vertical flipping.

 - The value predictions were being needlessly normalized to be between 0 and 1 during training 
 (and not denormalized during inference), causing the output of the network to only predict positive values 
 even for disadvantageous positions.

 - The reward discounting was being applied multiple times, effectively decreasing the range of rewards and therefore making the training loss seem lower

 - Initially forgot to include the outputs of the policy network in the UCT score computation, making that branch of the network completely useless in a truly spectacular display of stupidity

 - Accidentally moved the model from GPU to CPU at the end of the first round of training and kept it there for remaining rounds. Spent a long time wondering why the first round took one hour and the second took five.



## References
[Silver, David, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert et al. "Mastering the game of go without human knowledge." _nature_ 550, no. 7676 (2017): 354-359.](https://www.nature.com/articles/nature24270;)
