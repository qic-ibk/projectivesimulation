# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov and Katja Ried, implementing ideas from 

'Projective simulation for artificial intelligence'
Hans J. Briegel & Gemma De las Cuevas
Scientific Reports 2, Article number: 400 (2012) doi:10.1038/srep00400

and 

'Projective Simulation for Classical Learning Agents: A Comprehensive Investigation'
Julian Mautner, Adi Makmal, Daniel Manzano, Markus Tiersch & Hans J. Briegel
New Generation Computing, Volume 33, Issue 1, pp 69-114 (2015) doi:10.1007/s00354-015-0102-0
"""

import __future__
import numpy as np

class BasicPSAgent(object):
	"""Projective Simulation agent with two-layered network. Features: forgetting, glow, reflection, optional softmax rule. """
	
	def __init__(self, num_actions, num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections):
		"""Initialize the basic PS agent. Arguments: 
            - num_actions: integer >=1, 
            - num_percepts_list: list of integers >=1, not nested, representing the cardinality of each category/feature of percept space.
            - gamma_damping: float between 0 and 1, controls forgetting/damping of h-values
            - eta_glow_damping: float between 0 and 1, controls the damping of glow; setting this to 1 effectively switches off glow
            - policy_type: string, 'standard' or 'softmax'; toggles the rule used to compute probabilities from h-values
            - beta_softmax: float >=0, probabilities are proportional to exp(beta*h_value). If policy_type != 'softmax', then this is irrelevant.
            - num_reflections: integer >=0 setting how many times the agent reflects, ie potentially goes back to the percept. Setting this to zero effectively deactivates reflection.
            """
		
		self.num_actions = num_actions
		self.num_percepts_list = num_percepts_list
		self.gamma_damping = gamma_damping
		self.eta_glow_damping = eta_glow_damping
		self.policy_type = policy_type
		self.beta_softmax = beta_softmax
		self.num_reflections = num_reflections
		
		self.num_percepts = int(np.prod(np.array(self.num_percepts_list).astype(np.float64))) # total number of possible percepts
		
		self.h_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.float64) #Note: the first index specifies the action, the second index specifies the percept.
		self.g_matrix = np.zeros((self.num_actions, self.num_percepts), dtype=np.float64) #glow matrix, for processing delayed rewards
		
		if num_reflections > 0:
			self.last_percept_action = None  #stores the last realized percept-action pair for use with reflection. If reflection is deactivated, all necessary information is encoded in g_matrix.
			self.e_matrix = np.ones((self.num_actions, self.num_percepts), dtype=np.bool_) # emoticons
                #emoticons are initialized to True (happy, good choice) and set to false (sad, reflect again) only if the percept-action pair is used and does not yield a reward.
			
	def percept_preprocess(self, observation): # preparing for creating a percept
		"""Takes a multi-feature percept and reduces it to a single integer index.
        Input: list of integers >=0, of the same length as self.num_percept_list; 
            respecting the cardinality specified by num_percepts_list: observation[i]<num_percepts_list[i] (strictly)
            Output: single integer."""
		percept = 0
		for which_feature in range(len(observation)):
			percept += int(observation[which_feature] * np.prod(self.num_percepts_list[:which_feature]))
		return percept
		
	def deliberate_and_learn(self, observation, reward):
		"""Given an observation and a reward (from the previous interaction), this method
        updates the h_matrix, chooses the next action and records that choice in the g_matrix and last_percept_action.
        Arguments: 
            - observation: list of integers, as specified for percept_preprocess, 
            - reward: float
        Output: action, represented by a single integer index."""        
		self.h_matrix =  self.h_matrix - self.gamma_damping * (self.h_matrix - 1.) + self.g_matrix * reward # learning and forgetting
		if (self.num_reflections > 0) and (self.last_percept_action != None) and (reward <= 0): # reflection update
			self.e_matrix[self.last_percept_action] = 0
		percept = self.percept_preprocess(observation) 
		action = np.random.choice(self.num_actions, p=self.probability_distr(percept)) #deliberate once
		for i_counter in range(self.num_reflections):  #if num_reflection >=1, repeat deliberation if indicated
			if self.e_matrix[action, percept]:
				break
			action = np.random.choice(self.num_actions, p=self.probability_distr(percept))		
		self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
		self.g_matrix[action, percept] = 1 #record latest decision in g_matrix
		if self.num_reflections > 0:
			self.last_percept_action = action, percept	#record latest decision in last_percept_action
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
	