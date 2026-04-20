from config import load_args, load_args_species
from train import run_species

if __name__ == '__main__':
    args = load_args_species()
    run_species(args)