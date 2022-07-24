# Connect Four Agents
This repository contains a variety of agents capable of playing Connect Four, 
with skill levels ranging from trivial to extremely formidable. The available agents, 
in order of prowess (and approximate implementation complexity), are:
**RandomAgent** (moves randomly), **AlphaBetaAgent** (performs depth-limited tree search), 
**MCTSAgent** (performs full-depth Monte-Carlo simulations), and, most importantly, **AlphaZeroAgent** 
(based on the **AlphaZero** reinforcement learning algorithm from DeepMind).


The project exists mainly for the sake of exploring and comparing various gameplay strategies in a context 
which is reasonably (but not overwhelmingly) complex - however, there is also something to be said for the 
pleasure of playing against an algorithm that you actually understand.
Therefore, one could argue that this code sits squarely at the intersection of research and revelry. 
Feel free to use it for either or both.

## The Game Implementation
This repo contains a simple, no-frills implementation of the Connect Four game.
The current board state can be printed to the command line, and the game object exposes an interface for 
performing moves and checking whether the game has ended.
In particular, the game_over check has been highly optimized, in as much as Python allows, since it is performed at every move 
(and, for some of the agents, many thousands of times per user-facing move), which means it has the potential to be a major bottleneck.
If anyone else out there has spent as much time as I have profiling the performance of various Connect Four win-checking algorithms, you have my sympathy and my support.

## The Agents

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
but humans are rarely perfect, which keeps the game interesting. It would be 

### MCTSAgent
This agent follows the Monte Carlo Tree Search variant proposed in the AlphaZero paper, 
with the exception that, in the absence of an evaluation function, it uses a fixed prior distribution over moves 
(either uniform or gaussian, as with the RandomAgent), 
and performs full-depth simulations rather than using a value function as a short-circuit.
The agent then chooses the action corresponding to the highest visit count.
The MCTS search algorithm also forms the basis for the AlphaZero agent described below.

### AlphaZeroAgent
This agent is a from-scratch implementation (modulo PyTorch/PyTorch Lightning) of the AlphaZero reinforcement learning algorithm.


Each round of the training process consists of a fixed number of self-play games against a 
randomly selected previous version of the agent, after which the agent's network is retrained so that 
its policy estimates better match the action scores previously computed by the MCTS module, and the 
value estimates better match the actual outcomes of the games. 
After each round, the newly trained agent is evaluated in a series of games against the previous best agent - 
if the new agent wins by a minimum percentage (ex 55%), then it is checkpointed and becomes the new best agent 
(and is used to evaluate subsequent agents).
During self-play agents are able to resign if the value of the current state and all of its immediate children 
is below a particular threshold - this measure is intended simply to save computation. 
In addition to the exploration bonus used during MCTS simulations,
During self-play, exploration is encouraged in several ways: 
 - The actions during MCTS simulations are chosen according a score which includes an exploration bonus. 
 This bonus decays each time an action is chosen, meaning less-frequently selected actions will eventually be selected.
 - At each self-play move (after the MCTS simulations are complete), actions are chosen probabilistically 
based on MCTS scores, which are computed based on visit counts but controlled by a "temperature" parameter - 
a low temperature biases the agent toward the highest-scoring action,
whereas a high temperature increases the probability of low-scoring actions being selected.
 - The MCTS scores are further perturbed via a small amount of Dirichlet noise.


## Running the code
Users have the option of playing a two player game (with no AI algorithm on the other end), 
playing one-on-one against one of the agents, watching a game played between two agents, 
or training the AlphaZero agent from scratch.
For convencience, the repository includes a precomputed starter move tree for the MCTS agent 
(to speed up the computation for the first few moves), and a pre-trained neural network for the AlphaZero agent.

### Dependencies
 - PyTorch
 - PyTorch Lightning
 - Numpy

### Two Player Game
To simply play a game against another human, users can run `run_two_player.py` script (no command line arguments are needed).
Why anyone would want to use the command line interface instead of a physical board (or at least a GUI) is question mere mortals most likely cannot answer, 
but the functionality is there for those who want it.

### One Player Game
To play a 

- --agent

RandomAgent arguments
--gaussian (default = True)

AlphaBeta arguments


### Pitting Agents Against One Another


### Train AlphaZero

## References
