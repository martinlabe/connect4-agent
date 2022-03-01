import sys
import os
import shutil
import ray.rllib.agents.dqn as dqn
from gym_connect4.envs.connect4_env import Connect4Env
from src.model import Connect4Model
from time import sleep

## UTILS #####################################################################
models_path = "./models/"
check = False


def get_config(env):
    """generate the rllib config"""
    ## DQN
    config = dqn.DEFAULT_CONFIG.copy()
    config["num_atoms"] = 2
    ## Rainbow
    config["n_step"] = 5
    config["noisy"] = True
    config["num_atoms"] = 10
    config["v_min"] = -1
    config["v_max"] = 100
    config["double_q"] = True
    config["dueling"] = True
    # config["exploration"] = {}
    ## Env
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
    # config["train_batch_size"] = 500
    config["num_workers"] = 1 if check else 8
    config["render_env"] = False
    config["record_env"] = False
    config["num_gpus"] = 1
    return config


def load_weights(trainer):
    """load available weights"""
    # checking that the models directory exists
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    # if the models directory is empty
    if len(os.listdir(models_path)) == 0:
        print("No checkpoint. Starting from scratch.")
        return
    # load last checkpoint dir
    last_checkpoint_dir = max(os.listdir(models_path))
    last_checkpoint_num = int(last_checkpoint_dir[-6:])
    # load the corresponding weight
    last_checkpoint = sorted(os.listdir(models_path + last_checkpoint_dir))[1]
    trainer.restore(models_path + last_checkpoint_dir + '/' + last_checkpoint)
    print(f"Checkpoint {last_checkpoint_num} loaded.")


def get_trainer(env):
    """create the trainer with the config and load the weights"""
    config = get_config(env)
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
def train(env):
    """train the network - eventually load the previous weights"""
    trainer = get_trainer(env)
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
def print_winner(winner):
    res = "\n"
    if winner:
        res += "██    ██  ██████  ██    ██     ██     ██ ██ ███    ██ ██\n"
        res += " ██  ██  ██    ██ ██    ██     ██     ██ ██ ████   ██ ██\n"
        res += "  ████   ██    ██ ██    ██     ██  █  ██ ██ ██ ██  ██ ██\n"
        res += "   ██    ██    ██ ██    ██     ██ ███ ██ ██ ██  ██ ██   \n"
        res += "   ██     ██████   ██████       ███ ███  ██ ██   ████ ██\n"
    else:
        res += "██    ██  ██████  ██    ██     ██       ██████  ███████ ███████ ██\n"
        res += " ██  ██  ██    ██ ██    ██     ██      ██    ██ ██      ██      ██\n"
        res += "  ████   ██    ██ ██    ██     ██      ██    ██ ███████ █████   ██\n"
        res += "   ██    ██    ██ ██    ██     ██      ██    ██      ██ ██        \n"
        res += "   ██     ██████   ██████      ███████  ██████  ███████ ███████ ██\n"
    print(res)


def print_res(env):
    """print the results of the game"""
    if env.player == 1:
        if env.state == "WIN":
            print_winner(True)
        elif env.state == "WRONG":
            print("Vous avez fait un coup invalide.")
    else:
        if env.state == "WIN":
            print_winner(False)
        elif env.state == "WRONG":
            print("Votre adversaire a fait un coup invalide.")
    if env.state == "DRAW":
        print("Partie nulle.")


def play(env):
    """create the interface between the player and the agent"""
    trainer = get_trainer(env)
    trainer.config["exploration"] = False
    env.reset()
    done = {"__all__": False}
    sleep(1)
    while not done["__all__"]:
        env.render(mode="human")
        a = int(input("Which column do you play? (1-7)\n")) - 1
        obs, reward, done, info = env.step({1: a})
        if done["__all__"]:
            break
        b = trainer.compute_single_action(obs[2], explore=False, policy_id="player")
        obs, reward_b, done, info = env.step({2: b})
        if done["__all__"]:
            break
    print_res(env)
    return


## CLEAN #######################################################################
def clean():
    """clean the project files"""
    folder = './models/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


## CLEAN #######################################################################
def model(env):
    """print the model summary"""
    Connect4Model(myenv.observation_space,
                  myenv.action_space,
                  256,
                  get_config(myenv),
                  "Connect4CNN").base_model.summary()


## MAIN ########################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: wrong usage.\nPlease type 'python main.py help'")
        exit(2)

    if "--check" in sys.argv:
        check = True
    if "--quick" in sys.argv:
        check = False

    ENV_CONFIG = {
        "width": 7,
        "height": 6,
        "connect": 4,
        "verbose": check,
        "visualization": check
    }
    myenv = Connect4Env(ENV_CONFIG)

    if sys.argv[1] == "train":
        train(myenv)
    elif sys.argv[1] == "play":
        play(myenv)
    elif sys.argv[1] == "help":
        help()
    elif sys.argv[1] == "clean":
        clean()
    elif sys.argv[1] == "model":
        model(myenv)
    else:
        print("Error: command unknown.\nPlease type 'python main.py help'")
