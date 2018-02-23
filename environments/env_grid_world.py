# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Katja Ried, implementing ideas from 

'Projective simulation applied to the grid-world and the mountain-car problem'
Alexey A. Melnikov, Adi Makmal & Hans J. Briegel
arXiv:1405.5459 (2014)

and 

'Meta-learning within projective simulation'
Adi Makmal, Alexey A. Melnikov, Vedran Dunjko & Hans J. Briegel
IEEE Access 4, pp. 2110-2122 (2016) doi:10.1109/ACCESS.2016.2556579
"""

#This code requires the following packages
import numpy as np


class TaskEnvironment(object):
    """Grid world environment: a two-dimensional, discrete 'maze'
    which contains rewards in well-defined places and possibly walls that constrain 
    the agent's movement."""
    
    def __init__ (self, dimensions):
        """Given a list of two integers>=1 which specify x,y-extensions, 
        initialize a grid world. Simple example: env = TaskEnvironment([2,3])"""
        self.max_steps_per_trial = 10**6
        self.num_percepts_list = dimensions
        self.position = np.array([0, 0]) #keeps track of where the agent is located  
        self.rewards = np.zeros(dimensions)  #specifies the rewards (if any) located at each gridpoint
        self.rewards[-1, -1] = 1  #By default, only the furthest corner is rewarded
        self.act_list = [[+1,0],[0,+1],[-1,0],[0,-1]]  #hard-code which action indices correspond to which movements in terms of x,y changes
        # each position in the walls array contains a nested list of forbidden moves.
        #The first entry refers to forbidden x moves, the second to y.
        self.num_actions = len(self.act_list)
        self.walls=[[[[],[]] for ycoord in range(self.num_percepts_list[1])] for xcoord in range(self.num_percepts_list[0])]
        #initialize with boundary walls  
        # use world.walls[x][y]=[move1,move2] or .append(move) to update
        for xcoord in range(self.num_percepts_list[0]):
            self.walls[xcoord][0][1].append(-1)
            self.walls[xcoord][dimensions[1]-1][1].append(+1)
        for ycoord in range(self.num_percepts_list[1]):
            self.walls[0][ycoord][0].append(-1)
            self.walls[dimensions[0]-1][ycoord][0].append(+1)
    
    def reset(self):
        self.position = np.array([0, 0])
        return self.position
    
    def move(self,action_index):
        """Given the agent's action index (int 0-3), returns the new position, reward and trial_finished."""
        #test whether the action is permissible   
        action = self.act_list[action_index]
        posx, posy = self.position
        if not action[0] in self.walls[posx][posy][0]:
            self.position += np.array([action[0], 0])
        if not action[1] in self.walls[posx][posy][1]:
            self.position += np.array([0, action[1]]) 
        reward = self.rewards[self.position[0], self.position[1]]
        trial_finished = False
        if reward == 1:  #reset to origin to avoid agent hanging around target all the time
            self.position = np.array([0, 0])
            trial_finished = True
        return self.position, reward, trial_finished
        