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
python main.py [model|train|play|clean|help] [--quick|--check]
```

The targets are:
 - __model__: print the summary of the model used
 - __train__: create the network and train it.
 - __play__: create a game between the user and the agent.
 - __clean__: remove the weights saved in ./models/
 - __help__: target print the documentation.

The options are:
 - __--quick__: enable multi-workers and disable verbose and visualization
 - __--check__: enable one worker and disable verbose and visualization

## Tensorboard

To see the results of the training in TensorBoard type:
```
tensorboard --logdir=~/ray_results
```

## About

Author : Martin Labé \
Date : 02/2022