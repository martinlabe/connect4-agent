import sys
import os
import ray.rllib.agents.dqn as dqn
from gym_connect4.envs.connect4_env import Connect4Env
from src.model import Connect4Model

## UTILS #####################################################################
models_path = "./models/"

def get_config():
    """generate the rllib config"""
    ## DQN
    config = dqn.DEFAULT_CONFIG.copy()
    config["num_atoms"] = 2
    ## Rainbow
    # rainbow_config = dqn_config.copy()
    # rainbow_config["n_step"] = 5
    # rainbow_config["noisy"] = True
    # rainbow_config["num_atoms"] = 10
    config["v_min"] = -1
    config["v_max"] = 100
    config["double_q"] = True
    config["dueling"] = True
    # config["exploration"] = {}
    ## Env
    ENV_CONFIG = {
        "width": 7,
        "height": 6,
        "connect": 4,
        "verbose": False
    }
    env = Connect4Env(ENV_CONFIG)
    obs = env.reset()
    config["env_config"] = ENV_CONFIG
    ## Multi-agent
    ## https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    config["multiagent"] = {
        "policies": {
            "player": (None,
                       env.observation_space,
                       env.action_space,
                       {"gamma": 0.98}),
        },
        "policy_mapping_fn":
            lambda agent_id, episode, **kwargs:
            "player"
    }
    ## Model
    config["model"] = {
        "custom_model": Connect4Model,
        "custom_model_config": {}
    }
    ## Resources
    config["num_workers"] = 6
    config["render_env"] = False
    config["record_env"] = False
    config["num_gpus"] = 1
    return config

def load_weights(trainer):
    """load available weights"""
    # if the models directory is empty
    if len(os.listdir(models_path)) == 0:
        print("No checkpoint. Starting from scratch.")
    # load last checkpoint dir
    last_checkpoint_dir = max(os.listdir(models_path))
    last_checkpoint_num = int(last_checkpoint_dir[-6:])
    # load the corresponding weight
    last_checkpoint = sorted(os.listdir(models_path + last_checkpoint_dir))[1]
    trainer.restore(models_path + last_checkpoint_dir + '/' + last_checkpoint)
    print(f"Checkpoint {last_checkpoint_num} loaded.")

def get_trainer():
    """create the trainer with the config and load the weights"""
    config = get_config()
    trainer = dqn.DQNTrainer(config, env=Connect4Env)
    load_weights(trainer)
    return trainer


## HELP ######################################################################
def help():
    """print the help page of the program"""
    res = ""
    res += "## CONNECT4 AGENT ##\n"
    res += "usage: python main.py [train|play]"
    print(res)


## TRAIN #######################################################################
def train():
    """train the network - eventually load the previous weights"""
    trainer = get_trainer()
    num_iterations = 10000
    num_step = 50
    num_start = trainer.iteration
    for i in range(num_start + 1, num_iterations):
        trainer.train()
        if i % num_step == 0:
            trainer.save(models_path)
            print(f"Checkpoint {i} exported")
    trainer.stop()


## PLAY ########################################################################
def play():
    """create the interface between the player and the agent"""
    pass


## MAIN ########################################################################
if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "play":
            play()
        elif sys.argv[1] == "help":
            help()
        else:
            print("Error: command unknown.\nPlease type 'python main.py help'")
    else:
        print("Error: wrong usage.\nPlease type 'python main.py help'")
