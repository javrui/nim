<link rel="stylesheet" href="css/styles.css">

# Q-Learning-based Python script for training and playing the Nim game

![Banner](images/imagen_4_1920x150.jpg)

## Project Overview

This project is:

- A Python script implementing the Q-learning algorithm for reinforcement learning.
- Designed to both train an AI and let you play against it in the Nim game.
- Inspired by a project from my 2024 training course (HarvardX)["CS50's Introduction to Artificial Intelligence with Python (2024)](formacion_CS50AI.md), but fully object-oriented and documented using Google-style docstrings.
- Developed using AI assistants—ChatGPT o3-mini-high, ChatGPT 4o, and DeepSeek—to generate and compare code.
- Includes a visual analysis of the Q-table to explore how the AI "thinks."

## Why this project?

Q-learning is fascinating—it allows an agent to learn optimal actions in an unknown environment using rewards and penalties, all based on a simple Bellman equation. It mirrors how we learn in real life, doesn’t it? (Though in life, the rewards we choose to follow make all the difference.)

I also wanted to evaluate AI-generated code, comparing different models and testing their reliability in implementing my design.


## Nim game overview

Nim is a strategy game where two players take turns removing tokens from heaps on the game board. Each turn, a player removes any number of tokens from a single heap. The player who clears the board with their move wins.


## Design foundations

Key classes and their relation to reinforcement learning concepts:

- Move → Represents an RL action.
- Board → Represents an RL state.
- Learner → The Q-learning agent.
- NimGame → The RL environment.
- NimTrainerPlayer → Manages training and human vs. AI gameplay.

## UML Class Diagram

    (To be added)


## Design challenges

In Q-learning, the reward is immediate. But in Nim game, the reward (win/loss) comes after the opponent’s move, not the agent's move.
Q-learning is typically used for single-agent environments. In a two-player game like Nim, both the agent’s and the opponent’s actions influence the environment.

This creates unique challenges when designing the learning process.


