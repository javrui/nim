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

        return f"From heap {self.heap_index+1}, remove {self.stones} stones."

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
        _str = "Board (* = stones; · = removed stone)\n"
        for index, heap in enumerate(self.heaps):
            if index == chosen_move.heap_index:
                _str += f"{index+1}: {'*'*(heap-chosen_move.stones)}{'·'*chosen_move.stones}\n"
            else:
                _str += f"{index+1}: {'*'*heap}\n"
        print(_str)
        return _str

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
        epsilon_min: float = 0.1
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

        Args:
            board (Board): The board (state) before taking the move.
            move (Move): The move (action) taken.
            reward (float): The reward received after taking the move.
            next_board (Board): The board (state) after taking the move.
        """
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
        return self.board, reward, game_over

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

    def get_human_move(self)-> Move:
        """
        The human is prompted to select a move from a list of valid moves.

        Returns:
            Move: The selected move.
        """
        valid_moves: List[Move] = self.game.get_valid_moves()
        print("Valid moves:")
        for idx, move in enumerate(valid_moves):
            print(f"  {idx}: {move}")

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

    def load_q_table_if_present(self) -> None:
        """
        Load the Q-table from the file specified by q_table_filename if it exists.
        if self.q_table_filename is "", the function returns without doing anything, no q table is assumed to exist
        """
        if self.q_table_filename == "":
            return
        try:
            self.learner.load_q_table(self.q_table_filename)
            print(f"Loaded Q-table from '{self.q_table_filename}'. Skipping training.")
        except IOError as e:
            print(f"Could not load Q-table from '{self.q_table_filename}': {e}. Proceeding with training.")

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

        Modified to simulate an opponent move after the agent's move. The Q-value update now uses the board
        state resulting from the opponent's move (if the game was not already over) and adjusts the reward accordingly.

        Args:
            episodes (int): The number of training episodes.
        """
        # Check if a saved Q-table exists. If so, load it and skip training.
        if os.path.exists(self.q_table_filename):
            self.load_q_table_if_present()
            return

        # Proceed with training if no valid Q-table was loaded.
        for episode in range(episodes):
            self.game.reset()
            current_board = self.game.board
            while not self.game.is_game_over():
                # -Agent's Move
                agent_move = self.learner.choose_move(current_board, train_mode=True)
                next_board, reward, game_over = self.game.step(agent_move)

                # Simulates random opponent's move if game is not over
                if not game_over:
                    opponent_valid_moves = self.game.get_valid_moves()
                    # For the opponent, select a random valid move.
                    opponent_move = random.choice(opponent_valid_moves)
                    next_board_after_opponent, opponent_reward, game_over = self.game.step(opponent_move)
                    # If the opponent's move ends the game, it means the agent loses.
                    if game_over:
                        reward = -1.0  # Negative reward for losing.
                    else:
                        # If the game still continues, you might still penalize the agent slightly.
                        reward = -self.game.step_penalty
                    # Use the board state after the opponent's move for the Q-value update.
                    next_board = next_board_after_opponent
                # --------------------------------------------------------

                # Update the Q-value for the agent's move using the (possibly updated) next_board and reward.
                self.learner.update_Q_value(current_board, agent_move, reward, next_board)
                current_board = next_board

            self.learner.decay_epsilon()

        self.save_q_table()

    def play_against_human(self) -> None:
        """
        Play a game of Nim against a human player.

        Before the game starts, if the learner's Q-table is empty and a saved Q-table exists in the file
        specified by q_table_filename, it is loaded. Then, the human and the agent alternate moves.
        The agent uses an epsilon-greedy strategy to select moves. The game continues until completion, and the winner is announced.

        """
        # If no Q-table is loaded in the learner and a saved Q-table exists, load it.
        if not self.learner.q_table and os.path.exists(self.q_table_filename):
            self.load_q_table_if_present()

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
                print(f"\nBoard after your move:\n{self.game.board}")
                if self.game.is_game_over():
                    print("Congratulations! You win!")
                    return
            else:
                # Nim IA turn
                chosen_move = self.learner.choose_move(self.game.board, train_mode=False)
                print(f"\nNim AI moves: {chosen_move}\n")
                self.game.board.show_stones_move(chosen_move)
                self.game.apply_move(chosen_move)
                print(f"\nBoard after NIm AI move:\n{self.game.board}")
                if self.game.is_game_over():
                    print(f"\nBoard after move:\n{self.game.board}")
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

    # Initialize the game environment with a configurable step penalty.
    game: NimGame = NimGame(initial_heaps, step_penalty=0.01)

    # Initialize the Q-learning agent with default hyperparameters.
    learner: Learner = Learner()

    # Create the NimTrainerPlayer instance, providing no  Q-table filename to start training from scratch.
    trainer_player: NimTrainerPlayer = NimTrainerPlayer(game, learner, q_table_filename="")

    # Train the agent. If a saved Q-table exists, it will be loaded and training will be skipped.
    print("Training the agent...")
    trainer_player.train(episodes=10000)

    # Let the human play against the trained agent.
    print("\nNow, let's play against the trained agent!")
    trainer_player.play_against_human()

if __name__ == "__main__":
    main()
