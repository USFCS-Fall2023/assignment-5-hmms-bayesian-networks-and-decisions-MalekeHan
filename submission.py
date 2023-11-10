import random
import argparse
import codecs
import os
import numpy as np
from HMM import *

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='HMM Model Processing Script')
    parser.add_argument('model', help='The path to the trained model file (without extension)')
    parser.add_argument('--forward', help='Run the forward algorithm on the observations file')
    parser.add_argument('--viterbi', help='Run the Viterbi algorithm on the observations file')
    parser.add_argument('observations', nargs='?', help='The path to the observations file', default='')

    args = parser.parse_args()

    # Load the model
    hmm_model = HMM()
    hmm_model.load(args.model)

    if args.forward:
        # Run the forward algorithm if the flag is present
        with codecs.open(args.forward, 'r', 'utf-8') as f:
            observations = f.read().strip().split()
            final_state_prob = hmm_model.forward(observations)
            print(f"The probability of the observation sequence is: {final_state_prob}")

    if args.viterbi:
        # Run the Viterbi algorithm if the flag is present
        with codecs.open(args.viterbi, 'r', 'utf-8') as f:
            for line in f.readlines():
                observation = line.strip().split()
                if observation:  # Ensure observation is not empty
                    state_sequence = hmm_model.viterbi(observation)
                    print(f"Viterbi state sequence: {' '.join(state_sequence)}")
                else:
                    print("Empty observation sequence.")