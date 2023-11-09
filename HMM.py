

import random
import argparse
import codecs
import os
import numpy as np

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions
        self.states = []  # Initialize states as an empty list

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans) and emission (basename.emit) files, as well as the probabilities."""
        # Load transitions
        with open(f"{basename}.trans", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] not in self.transitions:
                    self.transitions[parts[0]] = {}
                self.transitions[parts[0]][parts[1]] = float(parts[2])
                if parts[0] != '#' and parts[0] not in self.states:  # Add state if it's not the start symbol and not already added
                    self.states.append(parts[0])

        # Load emissions
        with open(f"{basename}.emit", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] not in self.emissions:
                    self.emissions[parts[0]] = {}
                self.emissions[parts[0]][parts[1]] = float(parts[2])



   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        state = '#'
        stateseq = []
        outputseq = []

        for i in range(n):
            # Choose the next state based on the current state's transition probabilities
            next_state = np.random.choice(
                list(self.transitions[state].keys()),
                p=list(self.transitions[state].values())
            )
            stateseq.append(next_state)

            # Choose the next output based on the next state's emission probabilities
            next_output = np.random.choice(
                list(self.emissions[next_state].keys()),
                p=list(self.emissions[next_state].values())
            )
            outputseq.append(next_output)

            # Update the current state
            state = next_state

        return Observation(stateseq, outputseq)



    def forward(self, observations):
        """Implements the forward algorithm for an HMM."""
        F = np.zeros((len(self.states), len(observations)))  # Use self.states to define the size of the forward matrix
        start_probs = self.transitions['#']

        # Initialize the forward matrix with the start probabilities
        for i, state in enumerate(self.states):
            if observations[0] in self.emissions[state]:  # Check if the first observation is possible for the state
                F[i, 0] = start_probs.get(state, 0) * self.emissions[state].get(observations[0], 0)

        # Iterate over the rest of the observations
        for t in range(1, len(observations)):
            for s_to_idx, s_to in enumerate(self.states):
                for s_from_idx, s_from in enumerate(self.states):
                    if observations[t] in self.emissions[s_to]:  # Check if the observation is possible for the state
                        F[s_to_idx, t] += (F[s_from_idx, t-1] * self.transitions[s_from].get(s_to, 0) * self.emissions[s_to].get(observations[t], 0))

        # Probability of the observation sequence is the sum of the final column
        prob_obs_sequence = np.sum(F[:, -1])
        return prob_obs_sequence



    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
         # Initialize Viterbi matrix (probabilities) and backpointers
        V = np.zeros((len(self.states), len(observation)))
        backpointers = np.zeros((len(self.states), len(observation)), 'int')

        # Initialize first column of Viterbi matrix
        start_probs = self.transitions['#']
        for state in self.states:
            if observation[0] in self.emissions[state]:  # Check if the first observation is possible for the state
                V[self.states.index(state), 0] = start_probs.get(state, 0) * self.emissions[state].get(observation[0], 0)

        # Iterate over the rest of the observations
        for t in range(1, len(observation)):
            for s in self.states:
                # Find the maximum probability and the state that provides this max probability
                max_prob, best_state = max(
                    (V[self.states.index(s_prev), t-1] * self.transitions[s_prev].get(s, 0) * self.emissions[s].get(observation[t], 0), s_prev) 
                    for s_prev in self.states
                )
                V[self.states.index(s), t] = max_prob
                backpointers[self.states.index(s), t] = self.states.index(best_state)

        # Find the final state with maximum probability
        final_state = np.argmax(V[:, -1])

        # Follow the backpointers to find the best path
        best_path = [self.states[final_state]]
        for t in range(len(observation)-1, 0, -1):
            final_state = backpointers[final_state, t]
            best_path.insert(0, self.states[final_state])

        return best_path

def test_load():
    model = HMM()
    model.load('two_english')
    print("Transitions Loaded: ", model.transitions)
    print("Emissions Loaded: ", model.emissions)

#test_load()


def test_viterbi():
    # Initialize and load model
    model = HMM()
    model.load('partofspeech.browntags.trained')

    # Define a list of observations to test
    observations_to_test = [['DET', 'NOUN', 'VERB'], ['ADJ', 'NOUN', 'PUNCT']]

    # Run the Viterbi algorithm on predefined observations
    for observation in observations_to_test:
        state_sequence = model.viterbi(observation)
        print('Viterbi best state sequence for:', ' '.join(observation))
        print(' '.join(state_sequence))

def test_forward():
    # Initialize and load model
    model = HMM()
    model.load('partofspeech.browntags.trained')

    # Define a list of observations to test
    observations_to_test = [['the', 'quick', 'brown', 'fox'], ['jumps', 'over', 'the', 'lazy', 'dog']]

    # Run the Forward algorithm on predefined observations
    for observation in observations_to_test:
        prob_sequence = model.forward(observation)
        print('Forward probability for:', ' '.join(observation))
        print(prob_sequence)




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