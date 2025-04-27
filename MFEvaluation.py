import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pickle
import os

class MFEvaluation(object):
    """
    Class for evaluating model outputs against tweet and entity datasets, reconstructing text from token indices,
    and setting up mappings for entities, tweets, groups, and roles.
    """
    
    def __init__(self, predicted_heads, data_path):
        """
        Initializes the MFEvaluation object by loading data, reconstructing text for tweets and entities,
        and preparing empty dictionaries for labels, groups, and other mappings.
        
        Args:
            predicted_heads (any): Predicted heads information (not yet used during initialization).
            data_path (str): Path to the directory containing 'drail_data.pickle'.
        """
        # Load preprocessed data from pickle file
        [tweet_id2tokens, entity_id2tokens, _, word_dict, _, _, word2idx] = \
            pickle.load(open(os.path.join(data_path, "drail_data.pickle"), "rb"), encoding="latin1")
        
        # Reverse mapping: from token index to word
        idx2word = {v: k for k, v in word2idx.items()}
        
        # Map each entity ID to its full reconstructed text (joined tokens)
        self.entity2name = {k: " ".join(idx2word[v] for v in values) for k, values in entity_id2tokens.items()}
        
        # Map each tweet ID to its full reconstructed text (joined tokens)
        self.tweet2tokens = {k: " ".join(idx2word[v] for v in values) for k, values in tweet_id2tokens.items()}
        
        # Initialize empty dictionaries for data fed into DRaiL
        self.entity2label = {}      # Map from entity ID to its label
        self.entity2tweet = {}      # Map from entity ID to its corresponding tweet ID
        self.tweet2entities = {}    # Map from tweet ID to its associated entity IDs
        self.tweet2label = {}       # Map from tweet ID to its label
        self.tweet2ideo = {}        # Map from tweet ID to ideology information
        self.tweet2topic = {}       # Map from tweet ID to topic information
        self.role2mf = {}           # Map from role to MF (Meaning Frame or Mentions Frame)
        self.mf2roles = {}          # Map from MF to associated roles
        self.entity2group = {}      # Map from entity ID to group ID
        self.concept2tweet = {}     # Map from concept to associated tweets
        
        # Initialize an empty dictionary to map group IDs to readable group names
        self.group2names = {}
