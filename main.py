import sys
import ray.rllib.agents.dqn as dqn
from gym_connect4.envs.connect4_env import Connect4Env
from src.model import Connect4Model


## HELP ######################################################################
def print_help():
    """print the help page of the program"""
    res = ""
    res += "## CONNECT4 AGENT ##\n"
    res += "usage: python main.py [train|play]"
    print(res)

## TRAIN #######################################################################
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
    # rainbow_config["v_min"] = -1
    # rainbow_config["v_max"] = 100
    # rainbow_config["double_q"] = True
    # rainbow_config["dueling"] = True

    ## Env
    config["env_config"] = ENV_CONFIG

    ## Multi-agent
    ## https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    config["multiagent"] = {
        "policies": {
            "player": (None,
                       env.observation_space,
                       env.action_space,
                       {"gamma": 0.99}),
        },
        "policy_mapping_fn":
            lambda agent_id:
            "player"
    }

    ## Model
    config["model"] = {
        "custom_model": Connect4Model,
        "custom_model_config": {}
    }

    ## Resources
    config["num_workers"] = 2
    config["render_env"] = False
    config["record_env"] = False
    config["num_gpus"] = 1
    # config["framework"] = "tf"

    return config

def train(env):
    config = get_config()
    num_iterations = 5
    trainer = dqn.DQNTrainer(config, env=Connect4Env)
    for i in range(num_iterations):
        results = trainer.train()
        print(results)
    # export policy checkpoint sur le trainer
    trainer.stop()

## MAIN ########################################################################
if __name__ == "__main__":
    if len(sys.argv) == 2:

        ENV_CONFIG = {
            "width": 7,
            "height": 6,
            "connect": 4
        }
        env = Connect4Env(ENV_CONFIG)
        obs = env.reset()

        if sys.argv[1] == "train":
            train(env)
        elif sys.argv[1] == "play":
            print("the play target is not implemented yet")
        elif sys.argv[1] == "help":
            print_help()
        else:
            print("Error: command unknown.\nPlease type 'python main.py help'")
    else:
        print("Error: wrong usage.\nPlease type 'python main.py help'")
