

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
        with open(f"{basename}.trans", 'r') as f: # make a universal open script
            for line in f:
                parts = line.strip().split()
                if parts[0] not in self.transitions: # is the state already in the present?
                    self.transitions[parts[0]] = {}
                self.transitions[parts[0]][parts[1]] = float(parts[2]) # convert the string to a float and add it to the dict
                if parts[0] != '#' and parts[0] not in self.states:  # add only if not start/already added
                    self.states.append(parts[0])

        with open(f"{basename}.emit", 'r') as f: # state output prob
            for line in f:
                parts = line.strip().split()
                if parts[0] not in self.emissions: # check if state is in the dict
                    self.emissions[parts[0]] = {} # add the prob
                self.emissions[parts[0]][parts[1]] = float(parts[2])



   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        state = '#' # inital state
        stateseq = [] # used to store the state seq
        outputseq = [] #observed outputs

        for i in range(n):
            next_state = np.random.choice( #choose next state based on the a random probabilty from the trans -- use a weighted choice
                list(self.transitions[state].keys()),
                p=list(self.transitions[state].values()) # trans probabilites to the states
            )
            stateseq.append(next_state)

            
            next_output = np.random.choice( # randomly select an output based on probs
                list(self.emissions[next_state].keys()),
                p=list(self.emissions[next_state].values()) # emission probs for outputs
            )
            outputseq.append(next_output)

            
            state = next_state

        return Observation(stateseq, outputseq) # observation obj with state/output sequences



    def forward(self, observations):
        forward_matrix = np.zeros((len(self.states), len(observations)))  # use self.states to define the size of the forward matrix
        start_probs = self.transitions['#']

        
        for i, state in enumerate(self.states): # forward matrix with the start probabilities
            if observations[0] in self.emissions[state]:  # check if the first observation is possible for the state
                forward_matrix[i, 0] = start_probs.get(state, 0) * self.emissions[state].get(observations[0], 0)

        
        for t in range(1, len(observations)): # go through each observation
            for s_to_idx, s_to in enumerate(self.states):
                for s_from_idx, s_from in enumerate(self.states):
                    if observations[t] in self.emissions[s_to]:  # is the observation possible for the state ???
                        transition_probability = self.transitions[s_from].get(s_to, 0) #prob from previous state to current
                        emission_probability = self.emissions[s_to].get(observations[t], 0) #prob of emission based on current state
                        previous_probability = forward_matrix[s_from_idx, t-1] # get the last prob value from the matrix
                        forward_matrix[s_to_idx, t] += previous_probability * transition_probability * emission_probability # update current coordinate

        prob_obs_sequence = np.sum(forward_matrix[:, -1]) # probability is the last column -- memoization
        return prob_obs_sequence



    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
         # Initialize Viterbi matrix (probabilities) and backpointers
        viterbi_matrix = np.zeros((len(self.states), len(observation)))
        backpointers = np.zeros((len(self.states), len(observation)), 'int')

        start_probs = self.transitions['#']
        for state in self.states:
            if observation[0] in self.emissions[state]:  # Check if the first observation is possible for the state
                viterbi_matrix[self.states.index(state), 0] = start_probs.get(state, 0) * self.emissions[state].get(observation[0], 0)

       
        for t in range(1, len(observation)):
            
            for s in self.states: # each possible current state 's'
                
                max_probability_for_s = float('-inf')  
                best_previous_state_for_s = None  

                
                for s_prev in self.states:
                    
                    transition_probability = self.transitions[s_prev].get(s, 0) # probability of transitioning from s_prev to s
                    
                    emission_probability = self.emissions[s].get(observation[t], 0) # emission probability for the current observation from state 's'
                    
                    previous_viterbi_probability = viterbi_matrix[self.states.index(s_prev), t-1] #  viterbi probability for the previous state and time step
                    
                    viterbi_product = previous_viterbi_probability * transition_probability * emission_probability
                    
                    if viterbi_product > max_probability_for_s: # check for the greater valur
                        max_probability_for_s = viterbi_product
                        best_previous_state_for_s = s_prev

                viterbi_matrix[self.states.index(s), t] = max_probability_for_s
                
                backpointers[self.states.index(s), t] = self.states.index(best_previous_state_for_s)


        final_state = np.argmax(viterbi_matrix[:, -1]) # get the last state

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

