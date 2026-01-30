from config import load_args
from train import run

if __name__ == '__main__':
    args = load_args()
    run(args)