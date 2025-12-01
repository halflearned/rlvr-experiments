import torch
import argparse

# Just testing imports for now
from rlvr_experiments.inference import *

def main(args, kwargs):
    print("Inside inference entrypoint")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    args, kwargs = parser.parse_known_args()
    return args, kwargs


if __name__ == "__main__":
    args, kwargs = parse_args()
    main(args, kwargs)