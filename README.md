<link rel="stylesheet" href="css/styles.css">

# 2024 Python q-Learning Nim player

## What is this project about?

I found Q-learning fascinating. Using a simple (Bellman) equation, an agent learns optimal actions in an unknown environment through rewards and punishments in an iterative process. Much like how we learn in life, don't you think? (In real life, the reward we choose to guide our actions, makes the difference)

Completing a Nim game implementation in Python was one of the projects in the course [(HarvardX): "CS50's Introduction to Artificial Intelligence with Python (2024)](formacion_CS50AI.md). I decided to experiment further by redesigning and extending the code to visualize the learning process.

## Nim game description

Nim is a strategy game where two players take turns removing tokens from heaps on the game board. Each turn, a player removes any number of tokens from a single heap. The player who clears the board with their move wins.

## Foundations of design

Create a Simple Q-Learning library suitable for different problems solving (Nim, path finding, ..)
Q-Learning library should consider training and agent acting in problem solving.
Use library to code a Nim game trainer and player.

Derivate classes with names that reflect the specific concepts and terminology of the game:

- Action → Move
- State → BoardState
- QLearningAgent → NimPlayer
- Environment → NimGame

## Design challenges

In Q-learning, the reward is immediate. But in Nim game, the reward (game won or lost) comes after opponent move, not own move.
Q-learning is designed for single-agent environments. So in a two-player game, the agent's actions and the opponent's actions are both part of the environment


![Under Construction](images/under_construction.jpg)