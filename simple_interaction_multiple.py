# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Katja Ried.
"""

import __future__
import numpy as np
import os# for current directory
import sys # include paths to subfolders for agents and environments
sys.path.insert(0, 'agents')
sys.path.insert(0, 'environments')

"""This file illustrates the use of environments that can be populated by 
multiple agents, which can influence each others' experiences."""

"""Import and initialise an environment and an agent.
Different environments or agent types require different arguments, 
as specified in the docstring/help inside the respective files."""

#how many agents will interact simultaneously?
num_agents = 20

#environment
import env_locust
#invasion_game requires no additional arguments
env = env_locust.TaskEnvironment(num_agents,200,5) #world of length 100, with sensory range 5

#agent
import ps_agent_basic
#parameters for the agent - explanations can be found in the comments inside the agent file
gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections =  0.01, 1, 'standard', 1, 0
#initialise a list of agents, which will interact with the world in parallel
agent_list = []
for i_agent in range(num_agents):	
	agent_list.append(ps_agent_basic.BasicPSAgent(env.num_actions, env.num_percepts_list, gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections))


"""Initialise and run interaction"""

#set number of trials and maximum number of steps in each trial
num_trials = 100
max_steps_per_trial = 10 

#initialise a record of performance
learning_curve = np.zeros([num_trials, num_agents]) #This records the success of each agent, averaged over one trial, for all trials
reward_list = np.zeros(num_agents) #temporarily stores the most recent rewards earned by each agent
for i_trial in range(num_trials):
    reward_trial_list = np.zeros(num_agents) #additive counter of the total rewards earned during the current trial, by each agent separately
    next_observation = env.reset() #percept for a single agent, the one which is up next
    """Note that, in multi-agent environments, the method 'move' takes an agent_index as an argument, 
    along with a single action, and returns a single new percept for the next agent 
    along with the reward for the current one."""
    for t in range(max_steps_per_trial):
        for i_agent in range(num_agents):
            #This is where the heart of the interaction takes place
            action = agent_list[i_agent].deliberate_and_learn(next_observation, reward_list[i_agent])
            next_observation, reward_list[i_agent], done = env.move(i_agent, action)
            reward_trial_list[i_agent] += reward_list[i_agent]
            if done:
                break
        learning_curve[i_trial] = reward_trial_list/(t+1)


"""Return results"""
print(np.sum(learning_curve, axis=1)/num_agents) #This is the reward, averaged over each trial and all agents, as a function of the number of trials

#import matplotlib.pyplot as plt
#plt.plot(np.sum(learning_curve, axis=1)/num_agents)
