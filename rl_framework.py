#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov and Katja Ried
"""

import __future__
import numpy as np
import os # for current directory
import sys # for importing agents and environments
sys.path.insert(0, 'agents')
sys.path.insert(0, 'environments')

"""This file contains the basic functions that run.py draws on: initialisation of agents and environments
and their interaction, both for the case of a single agent at a time and multiple agents in the same environment."""

def EnvList():
	return 'Driver_Game', 'Invasion_Game', 'Grid_World', 'Mountain_Car', 'Locusts_Multiple', 'Neverending_Color', 'FrozenLake-v0', 'Acrobot-v1', 'Blackjack-v0', 'OffSwitchCartpole-v0', 'Pendulum-v0' # 'Go9x9-v0'

def AgentList():
	return 'PS-basic', 'PS-sparse', 'PS-flexible', 'PS-generalization'

def CreateEnvironment(env_name, env_config = None):
	"""Given a name (string) and an optional config argument, this returns an environment.
    Environments must have the following methods and attributes for later use:
        - method reset: no argument, returns a discretized_observation
        - method move: takes action as an argument and returns discretized_observation, reward, trial_finished
        - attrib num_actions: integer
        - attrib num_percepts_list: list of integers >=1, not nested, representing the cardinality of each category/feature of percept space
        - attrib max_steps_per_trial: integer; after this number of steps the environment returns trial_finished=True"""
	if env_name == 'Driver_Game':
		import env_driver_game
		env = env_driver_game.TaskEnvironment()         
	elif env_name == 'Invasion_Game': 
		import env_invasion_game
		env = env_invasion_game.TaskEnvironment()
	elif env_name == 'Neverending_Color': 
		import env_neverending_color
		num_actions, reward_value, max_num_trials = env_config
		env = env_neverending_color.TaskEnvironment(num_actions, reward_value, max_num_trials)
	elif env_name == 'Locusts_Multiple': 
		import env_locust
		num_agents, world_size, sensory_range = env_config
		env = env_locust.TaskEnvironment(num_agents, world_size, sensory_range)
	elif env_name == 'Grid_World':
		import env_grid_world
		dimensions = env_config
		env = env_grid_world.TaskEnvironment(dimensions)
	elif env_name == 'Mountain_Car':
		import env_mountain_car
		discretization_num_bins = env_config
		env = env_mountain_car.TaskEnvironment(discretization_num_bins)
	elif env_name in ('Acrobot-v1', 'CarRacing-v0', 'FrozenLake-v0', 'Go9x9-v0', 'Blackjack-v0', 'OffSwitchCartpole-v0', 'Pendulum-v0'):
		import env_openai
		discretization_num_bins = env_config
		env = env_openai.OpenAIEnvironment(openai_env_name=env_name, discretization_num_bins=discretization_num_bins)
	return env
		
def CreateAgent(agent_name, agent_config = None):
	"""Given a name (string) and an optional config argument, this returns an agent.
    Agents must have a single method, deliberate_and_learn, which takes as input an observation 
    (list of integers) and a reward (float) and returns an action (single integer index)."""
	if agent_name == 'PS-basic':
		import ps_agent_basic # import the basic PS agent
		agent = ps_agent_basic.BasicPSAgent(agent_config[0], agent_config[1], agent_config[2], agent_config[3], agent_config[4], agent_config[5], agent_config[6])
	elif agent_name == 'PS-sparse':
		import ps_agent_sparse # import the basic PS agent with sparse memory encoding
		agent = ps_agent_sparse.BasicPSAgent(agent_config[0], agent_config[1], agent_config[2], agent_config[3], agent_config[4], agent_config[5], agent_config[6])
	elif agent_name == 'PS-flexible':
		import ps_agent_flexible # import the flexible PS agent
		agent = ps_agent_flexible.FlexiblePSAgent(agent_config[0], agent_config[1], agent_config[2], agent_config[3], agent_config[4])
	elif agent_name == 'PS-generalization':
		import ps_agent_generalization # import the PS agent with generalization
		agent = ps_agent_generalization.PSAgent(agent_config[0], agent_config[1], agent_config[2], agent_config[3], agent_config[4], agent_config[5], agent_config[6], agent_config[7], agent_config[8], agent_config[9], agent_config[10])
	return agent

class Interaction(object):
	
	def __init__(self, agent, environment):
		"""Set up an interaction (which is not actually run yet). Arguments: 
            agent: object possessing a method deliberate_and_learn, which takes as arguments (discretized_observation, reward) and returns action;
            environment: object possessing the following two methods:
                reset: no argument, returns a discretized_observation
                move: takes action as an argument and returns discretized_observation, reward, done"""
		self.agent = agent
		self.env = environment
		
	def single_learning_life(self, num_trials, max_steps_per_trial):
		"""Train the agent over num_trials, allowing at most max_steps_per_trial 
        (ending the trial sooner if the environment returns done),
        and return an array containing the time-averaged reward from each trial."""
		learning_curve = np.zeros(num_trials)
		reward = 0 #temporarily stores the reward for the most recent action
		for i_trial in range(num_trials):
			reward_trial = 0 #additive counter of the total rewards earned during the current trial
			discretized_observation = self.env.reset()
			for t in range(max_steps_per_trial):
				discretized_observation, reward, done = self.single_interaction_step(discretized_observation, reward)
				reward_trial += reward
				if done:
					break
			learning_curve[i_trial] = float(reward_trial)/(t+1)
		return learning_curve
		
	def single_interaction_step(self, discretized_observation, reward):
		action = self.agent.deliberate_and_learn(discretized_observation, reward)
		return self.env.move(action)

class Interaction_Multiple(object):
    
	def __init__(self, agent_list, environment):
		"""Set up an interaction for multiple agents in parallel. Arguments: 
			agent_list: list of agents, which are objects possessing a method deliberate_and_learn, which takes as arguments (discretized_observation, reward) and returns action;
			environment: object possessing the following two methods:
			reset: no argument, returns a discretized_observation
			move: takes action as an argument and returns discretized_observation, reward, done"""
		self.agent_list = agent_list
		self.num_agents = len(agent_list)
		self.env = environment
        
	def single_learning_life(self, num_trials, max_steps_per_trial):
		"""Train all agents over num_trials, allowing at most max_steps_per_trial 
        (ending the trial sooner if the environment returns done),
        and return an array containing the time-averaged rewards (?) from each trial."""
		learning_curve = np.zeros([num_trials, self.num_agents])
		reward_list = np.zeros(self.num_agents) #temporarily stores the most recent rewards earned by each agent
		for i_trial in range(num_trials):
			reward_trial_list = np.zeros(self.num_agents) #additive counter of the total rewards earned during the current trial, by each agent separately
			next_observation = self.env.reset() #percept for a single agent, the one which is up next
			"""Memo: environments for multiple agents should 
                take num_agents as (one of the) initialization parameter(s). The method move should take
                an agent_index as a parameter, along with a single action, 
                and return a single new percept for the next agent along with the reward for the current one."""
			for t in range(max_steps_per_trial):
				for i_agent in range(self.num_agents):
					action = self.agent_list[i_agent].deliberate_and_learn(next_observation, reward_list[i_agent])
					next_observation, reward_list[i_agent], done = self.env.move(i_agent, action)
					reward_trial_list[i_agent] += reward_list[i_agent]
					if done:
						break
			learning_curve[i_trial] = reward_trial_list/(t+1)
		return learning_curve

