"""
This module implements a Nim game trainer and player with Q-Learning
reinforcement learning. Game winner: the player who removes the last stone(s).

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

    def get_all_possible_states(self) -> List[tuple]:
        """
        Generate all possible board states from self.heaps.

        Each state is a tuple of heap counts.
        The returned list is sorted by:
          1. Ascending total number of stones.
          2. For equal totals, states with more empty heaps come first.
          3. Natural (lexicographical) order of the state.
        """
        def _generate_states(heaps: List[int]) -> set:
            states = set()
            state = tuple(heaps)
            states.add(state)
            for i, pile in enumerate(heaps):
                # Try removing from 1 up to all stones in the i-th heap.
                for remove in range(1, pile + 1):
                    new_heaps = heaps.copy()
                    new_heaps[i] -= remove
                    new_state = tuple(new_heaps)
                    if new_state not in states:
                        states.update(_generate_states(new_heaps))
            return states

        all_states = _generate_states(self.heaps)
        sorted_states = sorted(
            all_states,
            key=lambda state: (
                sum(state),                   # Ascending total stones.
                -sum(1 for h in state if h == 0),  # More empty heaps first.
                state                         # Then natural (lexicographical) order.
            )
        )
        return sorted_states

    def OLD_get_all_possible_states(self) -> List[tuple]:
        """
        Generate all possible states from the current board configuration.

        Returns:
            List[tuple]: A list of all possible states, sorted by:
                        1. Total number of stones (ascending).
                        2. Number of empty heaps (descending).
        """
        def _generate_states(heaps: List[int]) -> set:
            """
            Recursively generate all possible states from the given heaps.

            Args:
                heaps (List[int]): The current heap configuration.

            Returns:
                set: A set of all possible states, represented as tuples.
            """
            states = set()
            states.add(tuple(heaps))  # Add the current state
            for i, pile in enumerate(heaps):  # Iterate over each heap
                for stones in range(1, pile + 1):  # Remove 1 to all stones from the heap
                    new_heaps = heaps.copy()  # Create a copy of the heaps
                    new_heaps[i] -= stones  # Remove stones from the heap
                    if new_heaps[i] < 0:  # Skip invalid states
                        continue
                    new_heaps_tuple = tuple(new_heaps)  # Convert to tuple (hashable for set)
                    if new_heaps_tuple not in states:  # Avoid duplicates
                        states.update(_generate_states(new_heaps))  # Recursively generate states
            return states

        # Generate all possible states
        all_states = _generate_states(self.heaps)

        # Sort the states:
        # 1. By total number of stones (ascending).
        # 2. By number of empty heaps (descending).
        sorted_states = sorted(all_states, key=lambda x: (sum(x), -sum(1 for pile in x if pile == 0)))

        return sorted_states

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

    def show_q_table(self, initial_board: list[int]) -> None:
        """
        Print the Q-table as a formatted table.
        Rows: board states (each represented as a list of heaps) sorted according to all possible states.
        Columns: moves as tuples (heap_index, stones) ordered by:
            - Grouping by stones (ascending)
            - Within each group, by heap index (ascending)

        For each cell:
          - If the board's heap (at the given index) has less stones than required by the move,
            the cell shows "--" (impossible combination).
          - If the move is possible and a Q-value is present:
                * If the Q-value is 0, show "0".
                * Otherwise, show the integer derived from the first two digits of the
                  decimal portion (i.e. int(q_value * 100)).
          - If no Q-value is present (key not in q_table) and the move is possible, the cell is left blank.
        """

        # Character for non possible board-move combinations and table borders.
        brick = '░'
        q_value_count = 0
        missing_q_values_count = 0

        # Generate all possible board states from the initial configuration.
        board_obj = Board(initial_board)
        all_board_states = board_obj.get_all_possible_states()[1:]

        # Determine all possible moves from the initial configuration.
        num_heaps = len(initial_board)
        moves_set = set()
        for heap_idx in range(num_heaps):
            # For each heap, possible moves are from removing 1 stone up to the maximum in that heap.
            for stones in range(1, initial_board[heap_idx] + 1):
                moves_set.add((heap_idx, stones))
        # Sort moves first by the number of stones (ascending) then by heap index (ascending).
        moves = sorted(moves_set, key=lambda m: (m[1], m[0]))

        # Build and print the header row.
        header = f"{'Board/moves':<11}" + " ".join(f"{f'({m[0]+1},{m[1]})':>8}" for m in moves)

        print(header)
        print("-" * len(header))

        # For each board state (row), print the board and then the Q-value for each move (column).
        for board in all_board_states:
            board_str = str(list(board))
            #row = f"{board_str:<13}"
            row = f"{board_str:<9}"

            for heap_idx, stones in moves:
                # If the board's available stones in the given heap are less than required, mark as impossible.
                if board[heap_idx] < stones:
                    cell = brick
                else:
                    key = (board, heap_idx, stones)
                    if key in self.q_table:
                        q_val = self.q_table[key]
                        q_value_count += 1
                        # If Q-value is exactly 0, display "0".
                        if q_val == 0:
                            cell = "0" # clearer than 0.0
                        else:
                            cell = f"{int(q_val * 100)}"
                    else:
                        cell = "___"
                        missing_q_values_count += 1
                row += f"{cell:>9}".replace(' ', brick)
            print(row + brick)
        print(f"\nq_values count/max possible q_values: {q_value_count}/{q_value_count+missing_q_values_count} ({100*q_value_count/(q_value_count+missing_q_values_count):.0f}%)\n")

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

        for episode in range(episodes):
            self.game.reset()
            # Randomly decide whether the agent goes first in this episode (affects full round ordering).
            agent_goes_first = random.choice([True, False])

            # Run full rounds until the game is over.
            while not self.game.is_game_over():
                if agent_goes_first:
                    # Agent makes a move.
                    board_state = self.game.board.copy()
                    agent_move = self.learner.choose_move(board_state, train_mode=True)
                    next_board, reward, game_over = self.game.step(agent_move)
                    if game_over:
                        #If the agent wins on its move, update and break.
                        self.learner.update_Q_value(board_state, agent_move, reward, next_board)
                        break
                    # Now the opponent makes a move.
                    opponent_move = random.choice(self.game.get_valid_moves())
                    next_board_after_opp, opp_reward, game_over = self.game.step(opponent_move)
                    if game_over:
                        # If the opponent wins, penalize the agent’s previous move.
                        self.learner.update_Q_value(board_state, agent_move, -1.0, next_board_after_opp)
                        break
                    # Update the Q-value for the agent’s move using the board state after the opponent’s move.
                    self.learner.update_Q_value(board_state, agent_move, reward, next_board_after_opp)

                else:
                    # Opponent goes first in the round.
                    opponent_move = random.choice(self.game.get_valid_moves())
                    _, _, game_over = self.game.step(opponent_move)
                    if game_over:
                        break
                    # Now it is the agent’s turn.
                    board_state = self.game.board.copy()
                    agent_move = self.learner.choose_move(board_state, train_mode=True)
                    next_board, reward, game_over = self.game.step(agent_move)
                    if game_over:
                        self.learner.update_Q_value(board_state, agent_move, reward, next_board)
                        break
                    # Opponent makes a move after the agent.
                    opponent_move = random.choice(self.game.get_valid_moves())
                    next_board_after_opp, opp_reward, game_over = self.game.step(opponent_move)
                    if game_over:
                        self.learner.update_Q_value(board_state, agent_move, -1.0, next_board_after_opp)
                        break
                    self.learner.update_Q_value(board_state, agent_move, reward, next_board_after_opp)

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
        #print("Your valid moves: ([heap]->stones : move nr.)")
        print("Your turn: Choose a move from the following valid moves:")
        for idx, move in enumerate(valid_moves):
            #print(f"  {idx}: {move}")
            print(f"To move {move}, type {idx}")

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
                input("Press key to see AI move...")
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
    # Define the hyperparameters for the Q-learning agent.
    hiperparameters = {"α": 0.5, "γ": 0.9, "ϵ": 1.0, "ϵ_decay": 0.995, "ϵ_min": 0.09}
    # Environment step penalty (negative reward for each non-winning move).
    step_penalty = 0.01
    # Number of training episodes.
    training_episodes = 10000

    # Show the initial configuration and training parameters.
    print(f"Nim initial board: {initial_heaps}")
    print(f"Training IA agent with {training_episodes} episodes.")
    print(f"Hiperparameters: {hiperparameters}\n")

    # Initialize the game environment with a configurable step penalty.
    game: NimGame = NimGame(initial_heaps, step_penalty)
    # Initialize the Q-learning agent with hyperparameters.
    learner: Learner = Learner(alpha = hiperparameters['α'],
                               gamma = hiperparameters['γ'],
                               epsilon = hiperparameters['ϵ'],
                               epsilon_decay = hiperparameters['ϵ_decay'],
                               epsilon_min = hiperparameters['ϵ_min'])

    # Create the NimTrainerPlayer instance, providing no  Q-table filename to start training from scratch.
    trainer_player: NimTrainerPlayer = NimTrainerPlayer(game, learner, q_table_filename="")
    # Train the agent. If a saved Q-table exists, it will be loaded and training will be skipped.
    trainer_player.train(training_episodes)

    # Show Q-Table:
    learner.show_q_table(initial_heaps)

    # Let the human play against the trained agent.
    print("\nNow, let's play against the trained agent!")
    trainer_player.play_against_human()

if __name__ == "__main__":
    main()
