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

import numpy as np

class TaskEnvironment(object):
	"""Invasion Game: in this game, the agent is faced with an invader trying to come in
    through one of num_actions doors. The invader (environment) first holds up a sign (percept)
    hinting which door it will attack. The agent must then choose an action (which door to guard).
    If action == percept, then the agent defends successfully and is therefore rewarded."""
	
	def __init__(self):
		self.num_actions = 2
		self.num_percepts_list = np.array([self.num_actions])
		self.next_state = np.array([np.random.randint(self.num_actions)]) #encodes where the attacker will go next, which is also the percept
		
	def reset(self):
		return self.next_state
		
	def move(self, action):
		if self.next_state == action:
			reward = 1
		else:
			reward = 0
		episode_finished = True
		self.next_state = np.array([np.random.randint(self.num_actions)]) # 'terminal state' is the next state
		return self.next_state, reward, episode_finished
		