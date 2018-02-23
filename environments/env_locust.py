# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Katja Ried, implementing ideas from

'Modelling collective motion based on the principle of agency'
Katja Ried, Thomas Muller & Hans J. Briegel
arXiv:1712.01334 (2017)
"""

import numpy as np


class TaskEnvironment(object):
    """This is a one-dimensional, circular world in which multiple agents move around.
    Percepts show agents the net movement of their close neighbours relative to themselves.
    Actions are turning or keeping going. Agents are rewarded for aligning themselves with their neighbours.
    This environment is used to study the collective motion of marching locusts.
    Reference: 'Modelling collective motion based on the principle of agency',
    Katja Ried, Thomas Muller and Hans J. Briegel, arXiv:1712.01334."""
    
    def __init__(self, num_agents, world_size, sensory_range):
        """Initializes a world. Arguments:
        num_agents (int>0) - number of agents
        world_size (int>0) - length of world; ends are identified (ie world is circular)
        sensory range (int>0) - how many steps away an agent can see others.
        Simple example: env = TaskEnvironment(5,40,4) (for 5 agents)
        max_num_trials, max_steps_per_trial = 20, 30 """
        self.num_agents = num_agents;
        self.world_size = world_size;
        self.sensory_range = sensory_range;
        
        self.num_actions = 2 #turn or keep going
        self.num_percepts_list = [5] 
        self.num_max_steps_per_trial = 10**9
        
        self.positions = np.random.randint(world_size,size=num_agents) #where each agent is
        #Note that multiple agents can occupy the same position - they do not collide.
        self.speeds = np.ndarray.tolist(np.random.choice([-1,1],num_agents)) #which way they are going
        #note that positions is an array whereas speeds is a list

    def get_neighbours(self,agent_index):
        """Determine indices of all agents within visual range including self."""
        focal_pos = self.positions[agent_index];
        neighbours = np.ndarray.tolist(np.where(dist_mod(self.positions,focal_pos,self.world_size)<self.sensory_range+1)[0]);
        return(neighbours)
    
    def net_rel_mvmt(self,agent_index):
        """Returns the net flow of all neighbours (excluding self),
        with sign indicating movement relative to orientation of focal agent."""
        neighbours = self.get_neighbours(agent_index)
        neighbours.remove(agent_index)
        return(self.speeds[agent_index]*sum([self.speeds[index] for index in neighbours]))
        
    def get_percept(self,agent_index):
        """Given an agent index, returns an integer [0,4] encoding the net flow relative to self (truncated at abs<=2)."""
        #compute percept
        net_rel_move = self.net_rel_mvmt(agent_index)
        #map to limited range of percepts
        if net_rel_move<-2:
            net_rel_move=-2
        if net_rel_move>+2:
            net_rel_move=2
        return(net_rel_move+2)

    def move(self,agent_index, action):
        """Given an agent_index and that agent's action (0 for turn, 1 for keep going),
        this function updates their speed and position and computes their reward,
        along with the percept for the next agent in the list."""
        self.speeds[agent_index] = self.speeds[agent_index]*(action*2-1)
        self.positions[agent_index] = np.remainder(self.positions[agent_index]+self.speeds[agent_index],self.world_size)
        reward = (np.sign(self.net_rel_mvmt(agent_index))+1)/2
        next_percept = self.get_percept((agent_index+1)%self.num_agents)
        return ([next_percept], reward, False)
    
    def reset(self):
        """Sets positions and speeds back to random values and returns the percept for the 0th agent."""
        self.positions = np.random.randint(self.world_size,size=self.num_agents)
        self.speeds = np.ndarray.tolist(np.random.choice([-1,1],self.num_agents))
        return([self.get_percept(0)])
        
def dist_mod(num1,num2,mod):
    """Distance between num1 and num2 (absolute value)
    if they are given modulo an integer mod, ie between zero and mod.
    Also works if num1 is an array (not a list) and num2 a number or vice versa."""
    diff=np.remainder(num1-num2,mod)
    diff=np.minimum(diff, mod-diff)
    return(diff)
