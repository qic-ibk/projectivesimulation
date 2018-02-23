# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov and Katja Ried, implementing ideas from 

'Projective simulation with generalization'
Alexey A. Melnikov, Adi Makmal, Vedran Dunjko & Hans J. Briegel
Scientific Reports 7, Article number: 14430 (2017) doi:10.1038/s41598-017-14740-y
"""

import numpy as np

class TaskEnvironment(object):
	"""Invasion game with never-ending colours: in this variant of the invasion game,
    the percept has two components: the first heralds which door the invader will attack
    (and therefore which action will earn a reward), while the second is a 'colour' that 
    is different in each trial, so that the agent never encounters the same percept twice."""
	
	def __init__(self, num_actions, reward_value, max_num_trials):
		"""Initialisation. Arguments:
			num_actions (int>0)
			reward_value (float): reward for correct action
			max_num_trials (int>0).
		Simple example: env = TaskEnvironment(2, 1, 100)
        	max_num_trials, num_agents = 100, 10 """
		self.num_actions = num_actions
		self.num_percepts_list = np.array([num_actions, max_num_trials])
		self.reward_value = reward_value
		self.next_state = np.array([np.random.randint(self.num_actions), 0])
		
	def reset(self):
		return self.next_state
		
	def move(self, action):
		if self.next_state[0] == action:
			reward = self.reward_value
		else:
			reward = 0
		episode_finished = True
		self.next_state = np.array([np.random.randint(self.num_actions), self.next_state[1]+1]) 
		return self.next_state, reward, episode_finished