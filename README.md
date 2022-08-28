# Yinsh_Game

The project aims to design an Agent that can play Yinsh games. [YINSH](https://en.wikipedia.org/wiki/YINSH) is an abstract strategy board game designed by game designer Kris Burm. It combines the rules of Othello and Gomoku. The player who removes three rings first will win, but as the rings decrease, the dominant side may be disadvantaged.

# Game Interface

The idea is to design a framework that can integrate different turn based multiplayer games.

## Guide

The python version supported is `python 3.8+`.
All required classes and methods to implement a new game are defined in the [template.py](template.py)

Some additional modules are required:

- The `func_timeout` module. Install with: pip3 install func_timeout
- The `tkinter` module. Install with: sudo apt-get install python3-tk

There are three give general agents that will work with any game under directory [agent/](agents/): [random.py](agents/random.py).
 [agents.random.py](agents/staff_team_random/random.py) and [timeout.py](agents/staff_team_random/timeout.py).


## Usage

The game can be run with specified runner. The only difference between runner is the first two line of the code (importing different game files.) The options can be found with following command:

```
python yinsh_runner.py -h
```

If running Sequence, note that the game will start in fullscreen mode. Press F11 to toggle fullscreen. The game's activity log now appears as a separate window.

## Feature

- save the print as log
- save replay
- play saved replay file
- run multiple games in sequential
