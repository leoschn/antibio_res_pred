from config import load_args, load_args_species
from train import run

if __name__ == '__main__':
    args = load_args_species()
    run(args)