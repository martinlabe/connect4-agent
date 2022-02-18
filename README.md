# Connect4 Agent

An agent trained with the Ray rllib to play the famous Connect4 game.

## Environment

To run the project please install the gym-interfaced Connect4Env available 
here :
> https://github.com/martinlabe/gym-connect4

And install the requirements :
```
pip install -r requirements.txt
```
## Usage
```
python main.py [train|play|help]
```

The targets are:
 - __train__: create the network and train it.
  If a network is available it will continue the training.
 - __play__: create a game between the user and the agent.
 - __help__: target print the documentation.

## About

Author : Martin Labé \
Date : 02/2022