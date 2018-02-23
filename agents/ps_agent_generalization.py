# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov, implementing ideas from 

'Projective simulation with generalization'
Alexey A. Melnikov, Adi Makmal, Vedran Dunjko & Hans J. Briegel
Scientific Reports 7, Article number: 14430 (2017) doi:10.1038/s41598-017-14740-y

and

'Projective simulation for artificial intelligence'
Hans J. Briegel & Gemma De las Cuevas
Scientific Reports 2, Article number: 400 (2012) doi:10.1038/srep00400
"""

import __future__
import numpy as np
from scipy.sparse import lil_matrix

class PSAgent(object):
	""" PS agent implementation """
	""" parameters: """
	""" num_actions - number of available actions, constant """
	""" num_percepts_list - number of percepts in each category, list of constants """
	""" generalization - True/False """
	""" ps_gamma, ps_eta - constants """
	""" policy_type - 'standard' or 'softmax' """
	""" ps_alpha - constant """
	""" hash_hash_bool - True/False """
	""" majority_vote - True/False """
	""" majority_time - specifies a trial, after which majorite voting is on """
	""" number_votes - number of votes in majority voting """
	
	def __init__(self, num_actions, num_percepts_list, generalization, gamma_damping, eta_glow_damping, policy_type, beta_softmax, hash_hash_bool, majority_vote, majority_time, number_votes):
		self.num_actions = num_actions
		self.num_categories = len(num_percepts_list)
		self.num_percepts_list = num_percepts_list
		self.n_percepts = np.prod(num_percepts_list)
		self.generalization = generalization
		self.hash_hash_bool = hash_hash_bool
		self.majority_vote = majority_vote
		self.majority_time = majority_time
		self.number_votes = number_votes
		
		self.gamma_damping = gamma_damping
		self.eta_glow_damping = eta_glow_damping
		self.policy_type = policy_type
		self.beta_softmax = beta_softmax
		
		self.time = 0 # internal time for majority voting
		
		if self.generalization:
			self.max_number_hash_clips = np.prod(np.array(self.num_percepts_list) + 1)
			self.adjacency_mat_dim = self.n_percepts + self.max_number_hash_clips + self.num_actions
			self.percept_encoding = np.zeros([self.n_percepts, self.num_categories])
			self.hash_clip_existence = np.zeros(np.array(num_percepts_list)+1, dtype=bool)
		else:
			self.adjacency_mat_dim = self.n_percepts + self.num_actions # constructs the basic two-layered network
			
		self.h_matrix = lil_matrix((self.adjacency_mat_dim, self.adjacency_mat_dim), dtype=np.float64)
		self.g_matrix = lil_matrix((self.adjacency_mat_dim, self.adjacency_mat_dim), dtype=np.float64)
	
	def percept_preprocess(self, observation): # preparing for creating a percept, creating generalizations
		percept = self.mapping_to_1d(observation)
		if np.sum(self.h_matrix[:, percept]) == 0: # percept is new
			if self.generalization:
				self.generalization_function(observation)
				self.percept_encoding[percept] = observation
			self.create_percept(observation, percept) # create new percept
		return percept
		
	def mapping_to_1d(self, observation):
		"""Takes a multi-feature percept and reduces it to a single integer index.
        Input: list of integers >=0, of the same length as self.num_percept_list; 
            respecting the cardinality specified by num_percepts_list: observation[i]<num_percepts_list[i] (strictly)
            Output: single integer."""
		percept = 0
		for which_feature in range(len(observation)):
			percept += int(observation[which_feature] * np.prod(self.num_percepts_list[:which_feature]))
		return percept
		
	def generalization_function(self, observation): # generalization function that compares all percepts, creates #-clips and edges
		for i_percept in range(self.n_percepts):
			if np.sum(self.h_matrix[:, i_percept]) != 0:
				difference_vector = observation - self.percept_encoding[i_percept]
				for i_element in range(self.num_categories):
					if difference_vector[i_element] == 0:
						difference_vector[i_element] = int(observation[i_element]+1) # equal element (+1 to avoid confusion with 0)
					else:
						difference_vector[i_element] = int(0) # hash is 0
				difference_vector = difference_vector.astype(int)
				if self.hash_hash_bool or (np.sum(difference_vector) != 0):
					if self.hash_clip_existence[tuple(difference_vector.reshape(1, -1)[0])] == False: # if #-clip is new
						self.hash_clip_creation(difference_vector, i_percept)
					
	def hash_clip_creation(self, difference_vector, i_percept): # create a #-clip and connect it
		hash_clip_encoding = self.mapping_to_1d(difference_vector)
		self.h_matrix[self.n_percepts + hash_clip_encoding, i_percept] = 1 # connect previous percept to this #-clip
		self.h_matrix[(self.adjacency_mat_dim-self.num_actions):self.adjacency_mat_dim, self.n_percepts + hash_clip_encoding] = 1 # connect to actions
		self.hash_clip_downwards(0, np.zeros(self.num_categories), difference_vector, False)
		self.hash_clip_upwards(0, np.zeros(self.num_categories), difference_vector)
		self.hash_clip_existence[tuple(difference_vector.reshape(1, -1)[0])] = True
		
	def hash_clip_downwards(self, i_position, vector, vector_orig, percept_bool):
		if i_position < (self.num_categories-1):
			for i_change in range(2):
				vector[i_position] = i_change * vector_orig[i_position]
				self.hash_clip_downwards(i_position + 1, vector, vector_orig, percept_bool)
		elif i_position == (self.num_categories-1):
			for i_change in range(2):
				vector[i_position] = i_change * vector_orig[i_position]
				vector = vector.astype(int)
				if self.hash_clip_existence[tuple(vector.reshape(1, -1)[0])]: # connecting two #-clips
					#print('downwards', vector_orig, vector)
					if percept_bool:
						hash_clip_1_encoding = self.mapping_to_1d(vector_orig-1)
						hash_clip_2_encoding = self.mapping_to_1d(vector)
						self.h_matrix[self.n_percepts + hash_clip_2_encoding, hash_clip_1_encoding] = 1
					else:
						hash_clip_1_encoding = self.mapping_to_1d(vector_orig)
						hash_clip_2_encoding = self.mapping_to_1d(vector)
						self.h_matrix[self.n_percepts + hash_clip_2_encoding, self.n_percepts + hash_clip_1_encoding] = 1
	
	def hash_clip_upwards(self, i_position, vector, vector_orig):
			if i_position < (self.num_categories-1):
				if vector_orig[i_position] == 0:
					for i_change in range(self.num_percepts_list[i_position] + 1):
						vector[i_position] = i_change
						self.hash_clip_upwards(i_position + 1, vector, vector_orig)
				else:
					vector[i_position] = vector_orig[i_position]
					self.hash_clip_upwards(i_position + 1, vector, vector_orig)
			elif i_position == (self.num_categories-1):
				if vector_orig[i_position] == 0:
					for i_change in range(self.num_percepts_list[i_position] + 1):
						vector[i_position] = i_change
						vector = vector.astype(int)
						if self.hash_clip_existence[tuple(vector.reshape(1, -1)[0])]: # connecting two #-clips
							hash_clip_1_encoding = self.mapping_to_1d(vector_orig)
							hash_clip_2_encoding = self.mapping_to_1d(vector)
							#print('upwards', vector_orig, vector)
							self.h_matrix[self.n_percepts + hash_clip_1_encoding, self.n_percepts + hash_clip_2_encoding] = 1
				else:
					vector[i_position] = vector_orig[i_position]
					vector = vector.astype(int)
					if self.hash_clip_existence[tuple(vector.reshape(1, -1)[0])]: # connecting two #-clips
						hash_clip_1_encoding = self.mapping_to_1d(vector_orig)
						hash_clip_2_encoding = self.mapping_to_1d(vector)
						#print('upwards', vector_orig, vector)
						self.h_matrix[self.n_percepts + hash_clip_1_encoding, self.n_percepts + hash_clip_2_encoding] = 1
	
	def create_percept(self, observation, percept_now):
		self.h_matrix[(self.adjacency_mat_dim-self.num_actions):self.adjacency_mat_dim, percept_now] = 1
		if self.generalization:
			self.hash_clip_downwards(0, np.zeros(self.num_categories), np.array(observation)+1, True)
		
	def delete_clip(self, clip_now):
		self.h_matrix[:, clip_now] = 0 # delete column
		self.h_matrix[clip_now, :] = 0 # delete row
	
	def policy(self, percept_now): # meta-policy, could be several random walks
		if self.majority_vote == False:
			return self.policy_general(percept_now)
		else: # majority voting part
			self.time += 1
			if self.time < self.majority_time:
				return self.policy_general(percept_now)
			else:
				p_vote = np.zeros(self.num_actions)
				for i_vote in range(self.number_votes):
					check_action = self.policy_general(percept_now)
					p_vote[check_action] += 1.0/self.number_votes
				major_action = np.argmax(p_vote)
				while check_action != major_action:
					check_action = self.policy_general(percept_now) 
				return major_action
		
	def deliberate_and_learn(self, observation, reward): # random walk
		if self.gamma_damping != 0:
			self.h_matrix = self.h_matrix.tocsc()
			self.h_matrix.data = (1 - self.gamma_damping) * self.h_matrix.data + self.gamma_damping # forgetting
			self.h_matrix = self.h_matrix.tolil()
		self.h_matrix += self.g_matrix * reward # learning
		
		clip_now = self.percept_preprocess(observation)
		self.g_matrix = (1 - self.eta_glow_damping) * self.g_matrix
		while clip_now < (self.adjacency_mat_dim - self.num_actions):
			clip_next = self.one_step_walk(clip_now)
			self.g_matrix[clip_next, clip_now] = 1
			clip_now = clip_next
		action = clip_now - (self.adjacency_mat_dim - self.num_actions)
		return action
			
	def one_step_walk(self, clip_now): # one step of a random walk
		if self.policy_type == 'standard':
			h_vector_now = self.h_matrix[:, clip_now]
			p_vector_now = ( h_vector_now / np.sum(h_vector_now) ).toarray().flatten()
		elif self.policy_type == 'softmax': 
			h_vector_now = self.beta_softmax * self.h_matrix[:, clip_now]
			h_vector_now = h_vector_now.toarray()
			for i_element in range(self.adjacency_mat_dim): # "decrease" the chance for nonexisting clips, temporary solution
				if h_vector_now[i_element] == 0:
					h_vector_now[i_element] = -10000
			h_vector_now_mod = h_vector_now - np.max(h_vector_now)
			p_vector_now = ( np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod)) ).flatten()
		clip_next = np.random.choice(p_vector_now.size, 1, p=p_vector_now)[0]
		return clip_next
		
	def clear_h_matrix(self): # deleting non-excisting clips from the matrices
		clear_indices = np.array([])
		for i_element in range(self.adjacency_mat_dim):
			if (np.sum(self.h_matrix[:, i_element]) == 0) and (np.sum(self.h_matrix[i_element, :]) == 0):
				clear_indices = np.append(clear_indices, [i_element])
		clear_matrix = self.h_matrix.toarray()
		clear_matrix = np.delete(clear_matrix, clear_indices, axis=0)
		clear_matrix = np.delete(clear_matrix, clear_indices, axis=1)
		return clear_matrix
		
	def clear_g_matrix(self): # deleting non-excisting clips from the matrices
		clear_indices = np.array([])
		for i_element in range(self.adjacency_mat_dim):
			if (np.sum(self.g_matrix[:, i_element]) == 0) and (np.sum(self.g_matrix[i_element, :]) == 0):
				clear_indices = np.append(clear_indices, [i_element])
		clear_matrix = self.g_matrix.toarray()
		clear_matrix = np.delete(clear_matrix, clear_indices, axis=0)
		clear_matrix = np.delete(clear_matrix, clear_indices, axis=1)
		return clear_matrix
		
	def h_matrix_output(self):
		return self.clear_h_matrix()
			
	def g_matrix_output(self):
		return self.clear_g_matrix()
		