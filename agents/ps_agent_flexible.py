# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Katja Ried, implementing ideas from 

'Projective simulation for artificial intelligence'
Hans J. Briegel & Gemma De las Cuevas
Scientific Reports 2, Article number: 400 (2012) doi:10.1038/srep00400

and 

'Projective Simulation for Classical Learning Agents: A Comprehensive Investigation'
Julian Mautner, Adi Makmal, Daniel Manzano, Markus Tiersch & Hans J. Briegel
New Generation Computing, Volume 33, Issue 1, pp 69-114 (2015) doi:10.1007/s00354-015-0102-0
"""

#This code requires the following packages
import __future__
import numpy as np

## PROJECTIVE SIMULATION AGENT (more sophisticated version)

class FlexiblePSAgent(object):
    """Projective Simulation agent with two-layered network and the distinctive feature that 
    it does not require one to specify the size or structure of percept space at initialization -- 
    instead, the agent maintains a dictionary of only those percepts it has encountered. 
    Note that this will only work if the percept space has been discretized beforehand,
    since otherwise the agent will (most likely) never encounter the same percept twice.
    Deliberation features: forgetting, glow, optional softmax rule. """

    def __init__(self, num_actions, gamma_damping, eta_glow_damping, policy_type, beta_softmax):
        """Initialize the basic PS agent. Arguments: 
        - num_actions: integer >=1, 
        - gamma_damping: float between 0 and 1, controls forgetting/damping of h-values
        - eta_glow_damping: float between 0 and 1, controls the damping of glow; setting this to 1 effectively switches off glow
        - policy_type: string, 'standard' or 'softmax'; toggles the rule used to compute probabilities from h-values
        - beta_softmax: float >=0, probabilities are proportional to exp(beta*h_value). If policy_type != 'softmax', then this is irrelevant.
        """
        self.num_percepts = 0
        self.num_actions = num_actions
        self.gamma_damping = gamma_damping  #damping parameter controls forgetting, gamma
        self.eta_glow_damping = eta_glow_damping   #damping of glow, eta
        self.policy_type = policy_type
        self.beta_softmax = beta_softmax

        self.h_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64)
        self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64)

        #dictionary of raw percepts 
        self.percept_dict = {}

    def percept_preprocess(self, observation): # preparing for creating a percept
        """Takes a percept of any immutable form -- numbers, strings or tuples thereof --
        or lists and arrays (which are flattened and converted to tuples), 
        checks whether it has been encountered before,
        updates num_percepts, percept_dict, h_matrix and g_matrix if required and
        and returns a single integer index corresponding to the percept."""
        #MEMO: the list of immutable types and the handling of mutable types (notably arrays) may need expanding
        # in order to ensure that we cover all 
        #Try to turn the observation into an immutable type
        if type(observation) in [str, int, bool, float, np.float64, tuple]: #This list should contain all relevant immutable types
            dict_key = observation
        elif type(observation) == list:
            dict_key = tuple(observation)
        elif type(observation) == np.ndarray:
            dict_key = tuple(observation.flatten())
        else:
            raise TypeError('Observation is of a type not supported as dictionary key. You may be able to add a way of handling this type.')
        
        if dict_key not in self.percept_dict:
            self.percept_dict[dict_key] = self.num_percepts
            self.num_percepts += 1
            #add column to hmatrix, gmatrix
            self.h_matrix = np.append(self.h_matrix,np.ones([self.num_actions,1]),axis=1)
            self.g_matrix = np.append(self.g_matrix,np.zeros([self.num_actions,1]),axis=1)

        return self.percept_dict[dict_key]
    
    def deliberate_and_learn(self, observation, reward):
        """Given an observation and a reward (from the previous interaction), this method
        updates the h_matrix, chooses the next action and records that choice in the g_matrix.
        Arguments: 
            - observation: any immutable object (as specified for percept_preprocess), 
            - reward: float
        Output: action, represented by a single integer index."""        
        self.h_matrix =  self.h_matrix - self.gamma_damping * (self.h_matrix - 1.) + self.g_matrix * reward # learning and forgetting
        percept = self.percept_preprocess(observation) 
        action = np.random.choice(self.num_actions, p=self.probability_distr(percept)) #deliberate once
        self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
        self.g_matrix[action, percept] = 1 #record latest decision in g_matrix
        return action	
		
    def probability_distr(self, percept):
        """Given a percept index, this method returns a probability distribution over actions
        (an array of length num_actions normalized to unit sum) computed according to policy_type."""        
        if self.policy_type == 'standard':
            h_vector = self.h_matrix[:, percept]
            probability_distr = h_vector / np.sum(h_vector)
        elif self.policy_type == 'softmax':
            h_vector = self.beta_softmax * self.h_matrix[:, percept]
            h_vector_mod = h_vector - np.max(h_vector)
            probability_distr = np.exp(h_vector_mod) / np.sum(np.exp(h_vector_mod))
        return probability_distr
