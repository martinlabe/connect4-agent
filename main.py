import sys
from gym_connect4.envs.connect4_env import Connect4Env


def print_help():
    """print the help page of the program"""
    res = ""
    res += "## CONNECT4 AGENT ##\n"
    res += "usage: python main.py [train|play]"


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
            print("the train target is not implemented yet")
        elif sys.argv[1] == "play":
            print("the play target is not implemented yet")
        elif sys.argv[1] == "help":
            print_help()
        else:
            print("Error: command unknown.\nPlease type 'python main.py help'")
    else:
        print("Error: wrong usage.\nPlease type 'python main.py help'")
