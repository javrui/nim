"""
This module implements a Nim game with Q-Learning reinforcement learning. The version of Nim considered is: the player who removes the last stones wins.

Running as a script:
    ...
Author:
    JRM 2025.01
License:
    GPL (GNU General Public License, Copyleft)
"""
import random
import pickle
import os
from typing import Dict, Tuple, List

class Move:
    """
    Represents a move in the Nim game.

    Attributes:
        heap_index (int): Index of the heap from which stones are removed.
        stones (int): Number of stones to remove from the heap.
    """

    def __init__(self, heap_index: int, stones: int) -> None:
        """
        Initialize a Move instance.

        Args:
            heap_index (int): Index of the heap from which stones are to be removed.
            stones (int): Number of stones to remove.
        """
        self.heap_index: int = heap_index
        self.stones: int = stones

    def __repr__(self) -> str:
        """
        Return a string representation of the move.

        Returns:
            str: A string in the format "Move(heap_index=X, stones=Y)"
                 heap_index starts by 1, for human readability.
        """
        return f"[{self.heap_index+1}]->{self.stones}"
        #return f"From heap {self.heap_index+1}, remove {self.stones} stones."

class Board:
    """
    Represents the board (state) for a game of Nim.

    Attributes:
        heaps (List[int]): A list representing the number of stones in each heap.
    """

    def __init__(self, heaps: List[int]) -> None:
        """
        Initialize the Board with the given heaps.

        The board will keep a copy of the initial heaps to allow resetting.

        Args:
            heaps (List[int]): A list where each element represents the initial number
                               of stones in a heap.
        """
        self.heaps: List[int] = heaps.copy()

    def apply_move(self, move: Move) -> None:
        """
        Apply a move to the board.

        This method updates the board by subtracting the number of stones specified in the move
        from the heap at the given index.

        Args:
            move (Move): The move to apply.

        Raises:
            ValueError: If the move is invalid (e.g., invalid heap index, non-positive stones,
                        or attempting to remove more stones than available in the heap).
        """
        if move.heap_index < 0 or move.heap_index >= len(self.heaps):
            raise ValueError("Invalid heap index in move.")
        if move.stones <= 0:
            raise ValueError("The number of stones to remove must be positive.")
        if self.heaps[move.heap_index] < move.stones:
            raise ValueError("Not enough stones in the selected heap to remove.")

        self.heaps[move.heap_index] -= move.stones

    def get_valid_moves(self) -> List[Move]:
        """
        Generate a list of all valid moves from the current board state.

        For each heap with at least one stone, a move is generated for every possible removal
        from 1 stone up to the number of stones in that heap.

        Returns:
            List[Move]: A list of Move objects representing all possible valid moves.
        """
        valid_moves: List[Move] = []
        for index, stone_count in enumerate(self.heaps):
            if stone_count > 0:
                for stones_to_remove in range(1, stone_count + 1):
                    valid_moves.append(Move(heap_index=index, stones=stones_to_remove))
        return valid_moves

    def is_game_over(self) -> bool:
        """
        Determine whether the game is over.

        The game is over when all heaps have zero stones.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return all(stones == 0 for stones in self.heaps)

    def __repr__(self) -> str:
        """
        Return a string representation of the board.

        Returns:
            str: A string in the format "Board(heaps=[...])".
        """
        _str = ""
        for index, heap in enumerate(self.heaps):
            _str += f"{index+1}: {'*'*heap}\n"

        return _str

    def show_stones_move(self, chosen_move: Move) -> str:
        """
        Prints a representation of the board and move. Also returns as string.

        Returns:
            str: A string where:
                moved stones are represented by '·'
                remaining stones are represented by '*'
        """
        _str = "Board (* = stones; · = just removed stone)\n"
        for index, heap in enumerate(self.heaps):
            if index == chosen_move.heap_index:
                _str += f"{index+1}: {'*'*(heap-chosen_move.stones)}{'·'*chosen_move.stones}\n"
            else:
                _str += f"{index+1}: {'*'*heap}\n"
        print(_str[:-1])
        return _str[:-1]

    def copy(self) -> 'Board':
        """
        Create a copy of the board.

        Returns:
            Board: A new Board instance with the same heaps.
        """
        return Board(self.heaps.copy())

    def generate_all_board_states(self):
        """
        Generates all possible board states from board configuration.

        Args:
            initial_board (list): The initial configuration of the board (e.g., [1, 3, 5]).

        Returns:
            set: A set of all possible board states, represented as tuples.
        """
        def _generate_states(board):
            states = set()
            states.add(tuple(board))  # Add the current state
            for i, heap in enumerate(board):  # Iterate over each pile with index and value
                for stones in range(1, heap + 1):  # Remove 1 to all stones from the pile
                    new_board = list(board)  # Create a copy of the board
                    new_board[i] -= stones  # Remove stones from the pile
                    if new_board[i] < 0:  # Skip invalid states
                        continue
                    new_board_tuple = tuple(new_board)  # Convert to tuple (hashable for set)
                    if new_board_tuple not in states:  # Avoid duplicates
                        states.update(_generate_states(new_board))  # Recursively generate states
            return states

        return _generate_states(self.heaps)



class Learner:
    """
    Represents a Q-learning based learner for the Nim game.

    Attributes:
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The current exploration rate.
        epsilon_decay (float): The decay factor for epsilon after each training episode.
        epsilon_min (float): The minimum exploration rate.
        q_table (Dict[Tuple[Tuple[int, ...], int, int], float]): The Q-values table mapping a board (state)
            and move (action) pair to its Q-value. The board is represented as a tuple of ints (heaps) and the
            move is represented as a tuple (heap_index, stones).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.01
    ) -> None:
        """
        Initialize the Learner with the specified Q-learning hyperparameters.

        Args:
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            epsilon (float): The initial exploration rate.
            epsilon_decay (float): The factor by which epsilon decays after each episode.
            epsilon_min (float): The minimum exploration rate.
        """
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.q_table: Dict[Tuple[Tuple[int, ...], int, int], float] = {}

    def _get_key(self, board: Board, move: Move) -> Tuple[Tuple[int, ...], int, int]:
        """
        Generate a key for the Q-table based on the board (state) and move (action).

        Args:
            board (Board): The current board (state) of the game.
            move (Move): The move (action) to be evaluated.

        Returns:
            Tuple[Tuple[int, ...], int, int]: A tuple key representing the board and move.
        """
        return (tuple(board.heaps), move.heap_index, move.stones)

    def get_Q_value(self, board: Board, move: Move) -> float:
        """
        Retrieve the Q-value for the given board (state) and move (action).

        Args:
            board (Board): The current board (state) of the game.
            move (Move): The move (action) for which the Q-value is requested.

        Returns:
            float: The Q-value for the board and move, or 0.0 if not present.
        """
        key: Tuple[Tuple[int, ...], int, int] = self._get_key(board, move)
        return self.q_table.get(key, 0.0)

    def set_Q_value(self, board: Board, move: Move, value: float) -> None:
        """
        Set the Q-value for the given board (state) and move (action).

        Args:
            board (Board): The current board (state) of the game.
            move (Move): The move (action) for which the Q-value is to be set.
            value (float): The new Q-value.
        """
        key: Tuple[Tuple[int, ...], int, int] = self._get_key(board, move)
        self.q_table[key] = value

    def choose_move(self, board: Board, train_mode: bool=True) -> Move:
        """
        Choose a move (action) for the given board (state) using an epsilon-greedy strategy.

        In train_mode, the exploration rate epsilon is used to determine whether to explore or exploit: with probability epsilon, a random valid move is selected (exploration).
        Otherwise, the move with the highest Q-value is selected (exploitation)
        if unique, a random among same highest q_value moves otherwise.

        Args:
            board (Board): The current board (state) of the game.

        Returns:
            Move: The selected move (action).

        Raises:
            ValueError: If no valid moves are available.
        """
        valid_moves: List[Move] = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available on the board.")

        if train_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Exploitation: choose the move with the highest Q-value.
        best_value: float = float("-inf")
        best_moves: List[Move] = []
        for move in valid_moves:
            q_value: float = self.get_Q_value(board, move)
            if q_value > best_value:
                best_value = q_value
                best_moves = [move]
            elif q_value == best_value:
                best_moves.append(move)
        return random.choice(best_moves)

    def update_Q_value(self, board: Board, move: Move, reward: float, next_board: Board) -> None:
        """
        Update the Q-value for the given board (state) and move (action) using the Q-learning update rule.

        The Q-learning update rule is defined as:
            Q(board, move) ← Q(board, move) + alpha * (reward + gamma * max_{move'} Q(next_board, move') - Q(board, move))

        Admits board as None, just returning.

        Args:
            board (Board): The board (state) before taking the move.
            move (Move): The move (action) taken.
            reward (float): The reward received after taking the move.
            next_board (Board): The board (state) after taking the move.
        """
        if board is None:
            return

        current_q: float = self.get_Q_value(board, move)
        if next_board.is_game_over():
            max_future_q: float = 0.0
        else:
            valid_moves: List[Move] = next_board.get_valid_moves()
            max_future_q = max(
                [self.get_Q_value(next_board, next_move) for next_move in valid_moves],
                default=0.0
            )
        new_q: float = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.set_Q_value(board, move, new_q)

    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate epsilon after each training episode.

        Ensures that epsilon does not fall below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename: str) -> None:
        """
        Save the Q-table to a file using pickle.

        Args:
            filename (str): The file path where the Q-table will be saved.

        Raises:
            IOError: If the file cannot be opened or written to.
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self.q_table, file)
        except OSError as e:
            raise IOError(f"Failed to save Q-table to {filename}: {e}") from e

    def load_q_table(self, filename: str) -> None:
        """
        Load the Q-table from a file.

        Args:
            filename (str): The file path from which to load the Q-table.

        Raises:
            IOError: If the file cannot be opened or read.
        """
        try:
            with open(filename, 'rb') as file:
                self.q_table = pickle.load(file)
        except OSError as e:
            raise IOError(f"Failed to load Q-table from {filename}: {e}") from e

    def show_q_table(self, initial_board: Board) -> None:
        """
        Print the Q-table as a formatted table.
        Rows: Board heaps (ordered as specified)
        Columns: Movements (ordered as specified)
        """
        # q_table complete possible states
        all_board_states = Board(initial_board).generate_all_board_states()

        # Extract all unique board states and moves
        # Sort by sum and inverse natural order
        board_states = sorted(set(state for state, _, _ in self.q_table.keys()),
                            key=lambda b: (sum(b), tuple(-x for x in b)))

        print(f"\nNo. of Values in q-Table: {len(self.q_table)}")
        print(f"No. present rows in q-Table:  {len(board_states)}")
        print(f"No. possible rows in q-table: {len(all_board_states)}\n")


        # Sort by stones first, then natural order
        moves = sorted(set((heap_index, stones) for _, heap_index, stones in self.q_table.keys()),
                    key=lambda m: (m[1], m))
        # Print header, 1-based index for heaps for human readability
        move_headers = [f"({h+1}, {s})" for h, s in moves]
        print(f"{'Board':<15} " + " ".join(f"{m:>8}" for m in move_headers))
        print("-" * (15 + 10 * len(move_headers)))

        # Print rows
        for board in board_states:
            board_str = str(list(board))

            if sum(1 for x in board if x > 0) == 1:
                board_mark = "*"
            elif sum(1 for x in board if x > 0) == 2:
                board_mark = "-"
            else:
                board_mark = " "
            board_str += board_mark
            #board_str = str(list(board)) + " *" if sum(1 for x in board if x > 0) == 1 else str(list(board)) + "  "
            row_values = [
                #f"{float(self.q_table.get((board, h, s), 0) * 100):>8.2f}"
                f"{float(self.q_table.get((board, h, s), 0) * 100):>8.0f}"
                if self.q_table.get((board, h, s), 0) != 0
                else " " * 8 for h, s in moves]

            #row_values = [f"{int(self.q_table.get((board, h, s), 0) * 100):>8d}" if self.q_table.get((board, h, s), 0) != 0 else " " * 8 for h, s in moves]
            print(f"{board_str:<13} " + " ".join(row_values))

class NimGame:
    """
    Represents the Nim game environment.

    In this version of Nim, the player who removes the last stone wins.
    The game is modeled as a reinforcement learning environment where the board
    represents the current state and a move represents an action.

    Attributes:
        initial_heaps (List[int]): A list representing the initial configuration of heaps.
        board (Board): The current board (state) of the game.
        step_penalty (float): The penalty applied for each move that does not win the game.
                              This value is used to encourage faster wins.
    """

    def __init__(self, initial_heaps: List[int], step_penalty: float = 0.01) -> None:
        """
        Initialize the NimGame with the given initial heaps and step penalty.

        Args:
            initial_heaps (List[int]): A list where each element represents the initial
                number of stones in a heap.
            step_penalty (float, optional): The penalty applied to non-winning moves.
                Defaults to -0.01.
        """
        self.initial_heaps: List[int] = initial_heaps.copy()
        self.board: Board = Board(initial_heaps)
        self.step_penalty: float = step_penalty

    def get_valid_moves(self) -> List[Move]:
        """
        Get all valid moves (actions) from the current board (state).

        Returns:
            List[Move]: A list of Move objects representing all possible valid moves.
        """
        return self.board.get_valid_moves()

    def apply_move(self, move: Move) -> None:
        """
        Apply a move (action) to the current board (state).

        Args:
            move (Move): The move (action) to apply.
        """
        self.board.apply_move(move)

    def is_game_over(self) -> bool:
        """
        Determine whether the game is over.

        Returns:
            bool: True if the game is over (i.e., all heaps have zero stones), False otherwise.
        """
        return self.board.is_game_over()

    def reset(self) -> None:
        """
        Reset the game to its initial configuration.
        """
        self.board = Board(self.initial_heaps)

    def step(self, move: Move) -> Tuple[Board, float, bool]:
        """
        Apply a move (action) to the board (state) and return the new board, the reward, and a flag indicating
        whether the game is over.

        A penalty (defined by the step_penalty parameter) is applied for each move to encourage faster wins.
        If the move results in a winning state (i.e., the board becomes game over), a positive reward is returned.

        Args:
            move (Move): The move (action) to apply.

        Returns:
            Tuple[Board, float, bool]: A tuple containing:
                - Board: The new board (state) after the move.
                - float: The reward obtained after the move. A reward of 1.0 is given if the move wins the game;
                  otherwise, the step_penalty is applied.
                - bool: True if the game is over after the move, False otherwise.
        """
        self.board.apply_move(move)
        game_over: bool = self.board.is_game_over()
        if game_over:
            reward: float = 1.0  # Positive reward for winning
        else:
            reward: float = -self.step_penalty  # Use the parameterized step penalty
        return self.board.copy(), reward, game_over

class NimTrainerPlayer:
    """
    Represents the Nim game trainer and player that uses Q-learning.

    Attributes:
        game (NimGame): The Nim game environment.
        learner (Learner): The Q-learning agent.
        q_table_filename (str): The file path used to load/save the Q-table.
    """

    def __init__(self, game: NimGame, learner: Learner, q_table_filename: str) -> None:
        """
        Initialize the NimTrainerPlayer with a game, a learner, and a Q-table filename.

        Args:
            game (NimGame): The Nim game environment.
            learner (Learner): The Q-learning agent.
            q_table_filename (str, optional): The file path for loading and saving the Q-table.
        """
        self.game: NimGame = game
        self.learner: Learner = learner
        self.q_table_filename: str = q_table_filename

    def load_q_table(self) -> bool:
        """
        Load the Q-table from the file specified by q_table_filename if it exists.
        if self.q_table_filename is "", the function returns without doing anything, no q table is assumed to exist

        Returns:
            bool: True if the Q-table was loaded successfully, False otherwise.
        """
        if self.q_table_filename == "" or not os.path.exists(self.q_table_filename):
            return False
        try:
            self.learner.load_q_table(self.q_table_filename)
            print(f"Loaded Q-table from '{self.q_table_filename}'. Skipping training.")
            return True
        except IOError as e:
            print(f"Could not load Q-table from '{self.q_table_filename}': {e}. Proceeding with training.")
            return False

    def save_q_table(self) -> None:
        """ Save the Q-table to the file specified by q_table_filename.
        """
        if self.q_table_filename == "":
            return

        self.learner.save_q_table(self.q_table_filename)
        print(f"Q-table saved to '{self.q_table_filename}'.")

    def train(self, episodes: int = 10000) -> None:
        """
        Train the Q-learning agent over a number of episodes. If a saved Q-table is found in the file
        specified by q_table_filename, it is loaded and training is skipped; otherwise, training is performed
        and the resulting Q-table is saved to that file.

        Two agents play against each other, the agent and a random opponent (opponent that chooses random moves).

        Args:
            episodes (int): The number of training episodes.
        """
        # If a saved Q-table exists, load it and skip training.
        if self.load_q_table():
            return

        # Proceed with training if no valid Q-table was loaded.
        for episode in range(episodes):
            self.game.reset()
            #previous_agent_board = None # Board state before the agent's move
            #previous_agent_move = None # move chosen by previous agent
            learning_agent_turn = True # Learning agent is playing (not random opponent)

            # Training loop:
            while True:
                board = self.game.board.copy()
                if learning_agent_turn:
                    move = self.learner.choose_move(board, train_mode=True)
                else:
                    # Opponent agent does not use q-table, but random moves
                    move = random.choice(self.game.get_valid_moves())

                next_board, reward, game_over = self.game.step(move)
                #if not learning_agent_turn and reward == 1:
                    # If the opponent wins, the agent loses.
                    #self.learner.update_Q_value(previous_agent_board, move, -1, board)

                # Update the Q-value for the agent's move
                if learning_agent_turn:
                    self.learner.update_Q_value(board, move, reward, next_board)
                    # to be used in case next (opponent) move wins game
                    #previous_agent_board = board.copy()
                    #previous_agent_move = move

                if game_over:
                    break

                learning_agent_turn = not learning_agent_turn

            # Decay epsilon after each episode.
            self.learner.decay_epsilon()

        self.save_q_table()

    def OLD_train(self, episodes: int = 10000) -> None:
        """
        Train the Q-learning agent over a number of episodes. If a saved Q-table is found in the file
        specified by q_table_filename, it is loaded and training is skipped; otherwise, training is performed
        and the resulting Q-table is saved to that file.

        Modified to simulate an opponent move after the agent's move. The Q-value update now uses the board
        state resulting from the opponent's move (if the game was not already over) and adjusts the reward accordingly.

        Args:
            episodes (int): The number of training episodes.
        """
        # Check if a saved Q-table exists. If so, load it and skip training.
        if os.path.exists(self.q_table_filename):
            self.load_q_table()
            return

        # Proceed with training if no valid Q-table was loaded.
        for episode in range(episodes):
            self.game.reset()
            current_board = self.game.board.copy()
            while not self.game.is_game_over():
                # -Agent's Move
                agent_move = self.learner.choose_move(current_board, train_mode=True)
                next_board, reward, game_over = self.game.step(agent_move)

                if not game_over:
                    # Simulates random opponent's valid move
                    opponent_valid_moves = self.game.get_valid_moves()
                    opponent_move = random.choice(opponent_valid_moves)
                    next_board_after_opponent, opponent_reward, game_over = self.game.step(opponent_move)

                    if game_over:
                        # If the opponent's move ends the game, it means the opponent wins
                        reward = -1.0
                    else:
                        # If the game still continues, penalize the agent slightly.
                        reward = -self.game.step_penalty
                #else:
                    # If the opponent's move ends the game, it means the agent loses.
                    #reward = -1.0
                    #else:
                        # If the game still continues, penalize the agent slightly.
                        #reward = -self.game.step_penalty
                    # Use the board state after the opponent's move for the Q-value update.
                    # next_board = next_board_after_opponent
                # --------------------------------------------------------

                # Update the Q-value for the agent's move using the (possibly updated) next_board and reward.
                self.learner.update_Q_value(current_board, agent_move, reward, next_board)
                if not game_over:
                    # Move to the next state after opponent's move
                    current_board = next_board_after_opponent
                else:
                    break

            self.learner.decay_epsilon()

        self.save_q_table()


    def get_human_move(self)-> Move:
        """
        The human is prompted to select a move from a list of valid moves.

        Returns:
            Move: The selected move.
        """
        valid_moves: List[Move] = self.game.get_valid_moves()
        print(50*'-')
        print("Your valid moves: ([heap]->stones : move nr.)")
        for idx, move in enumerate(valid_moves):
            #print(f"  {idx}: {move}")
            print(f"  {move} : {idx}")

        choice = None
        while (choice is None):
            try:
                choice: int = int(input("Enter the number of your chosen move: "))
                if choice < 0 or choice >= len(valid_moves):
                    print("Invalid choice. Try again.")
                    choice = None
                    continue
                chosen_move: Move = valid_moves[choice]
            except ValueError:
                print("Invalid input. Please enter an integer.")
                choice = None
                continue
        print()
        return chosen_move

    def play_against_human(self) -> None:
        """
        Play a game of Nim against a human player.

        Before the game starts, if the learner's Q-table is empty and a saved Q-table exists in the file
        specified by q_table_filename, it is loaded. Then, the human and the agent alternate moves.
        The agent uses an epsilon-greedy strategy to select moves. The game continues until completion, and the winner is announced.

        """
        # If no Q-table is loaded in the learner and a saved Q-table exists, load it.
        if not self.learner.q_table and os.path.exists(self.q_table_filename):
            self.load_q_table()

        first: str = input("Do you want to go first? (y/n): ").strip().lower()
        human_turn: bool = (first == 'y')
        self.game.reset()
        print(f"\nInitial Nim board:\n{self.game.board}")

        while not self.game.is_game_over():
            if human_turn:
                # Human turn
                chosen_move = self.get_human_move()
                self.game.board.show_stones_move(chosen_move)
                self.game.apply_move(chosen_move)
                # print(f"\nBoard after your move:\n{self.game.board}")
                if self.game.is_game_over():
                    print("Congratulations! You win!")
                    return
            else:
                # Nim IA turn
                chosen_move = self.learner.choose_move(self.game.board, train_mode=False)
                print(50*'-')
                print(f"Nim AI moves: {chosen_move}\n")
                self.game.board.show_stones_move(chosen_move)
                self.game.apply_move(chosen_move)
                #print(f"\nBoard after Nim AI move:\n{self.game.board}")
                if self.game.is_game_over():
                    print("Agent wins!")
                    return
            human_turn = not human_turn
            #print(f"Board after move:\n{self.game.board}")

def main() -> None:
    """
    Main function to run the Nim game training and play against a human.

    The function initializes the game environment with a default configuration,
    instantiates a Learner and a NimTrainerPlayer with a specified Q-table filename,
    trains the Q-learning agent (loading a saved Q-table if available), and then allows a
    human to play against the trained agent.
    """
    # Define the initial heaps configuration (e.g., heaps with 1, 3, and 5 stones).
    initial_heaps: List[int] = [1, 3, 5]
    #initial_heaps: List[int] = [2, 0, 0]


    # Initialize the game environment with a configurable step penalty.
    game: NimGame = NimGame(initial_heaps, step_penalty=0.01)

    # Initialize the Q-learning agent with default hyperparameters.
    learner: Learner = Learner()

    # Create the NimTrainerPlayer instance, providing no  Q-table filename to start training from scratch.
    trainer_player: NimTrainerPlayer = NimTrainerPlayer(game, learner, q_table_filename="")

    # Train the agent. If a saved Q-table exists, it will be loaded and training will be skipped.
    training_episodes = 10000
    print(f"Training the agent with {training_episodes} episodes.")
    trainer_player.train(training_episodes)

    # Show Q-Table:
    learner.show_q_table(initial_heaps)
    exit()

    # Let the human play against the trained agent.
    print("\nNow, let's play against the trained agent!")
    trainer_player.play_against_human()

if __name__ == "__main__":
    main()
