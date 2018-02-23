# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov, implementing ideas from 

'Projective simulation with generalization'
Alexey A. Melnikov, Adi Makmal, Vedran Dunjko & Hans J. Briegel
Scientific Reports 7, Article number: 14430 (2017) doi:10.1038/s41598-017-14740-y
"""

import numpy as np

class TaskEnvironment(object):
	"""Driver scenario implementation"""
	
	def __init__(self):
		self.num_actions = 2
		self.num_directions = 2
		self.num_colors = 2
		self.num_percepts_list = np.array([self.num_directions, self.num_colors])
		self.next_state = np.array([np.random.randint(self.num_directions), np.random.randint(self.num_colors)])
		
	def reset(self):
		return self.next_state
		
	def move(self, action):
		if (self.next_state[0] == action):
			reward = 1
		else:
			reward = 0
		episode_finished = True
		self.next_state = np.array([np.random.randint(self.num_directions), np.random.randint(self.num_colors)]) # 'terminal state' is the next state
		return self.next_state, reward, episode_finished