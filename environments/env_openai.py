# -*- coding: utf-8 -*-
"""
Copyright 2018 Alexey Melnikov and Katja Ried.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Alexey Melnikov and Katja Ried.

Reference:
'OpenAI Gym'
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang & Wojciech Zaremba
arXiv:1606.01540 (2016)
"""

import numpy as np
import gym # import from Gym

class OpenAIEnvironment(object):
    """This class serves as an interface between the standardized environments of OpenAI Gym and the PS agent.
    You must install the gym package and make sure that the programm can access it (ie provide the path) to run this code."""    
    
## NEED TO SET UP A NUMBER OF METHODS FOR INTERNAL USE FIRST

    def decompose_variable_base(self, total, base_list):
        """Given an integer total (between 0 and np.prod(base_list)-1)
        and a list base_list of integers >=0, this function returns a list coefs of integers in the interval 0 <= coefs[i] < base_list[i] such that
        total = sum( [ coefs[position]*np.prod(base_list[:position]) for position in range(len(base_list)) ] ).
        If base_list = [B]*length, this yields the representation of total in the base B 
        (with the i-th entry in the lists multiplying B**i, ie the ordering is the opposite of the usual way of writing multi-digit numbers)."""
        if total >= np.prod(base_list):
            raise Exception('Number too large for decomposition in given base.')
        remainder = total
        coefs=[]
        for position in range(len(base_list)-1):
            coefs.append( int( remainder % base_list[position] ) )
            remainder = (remainder - coefs[-1]) / base_list[position]
        coefs.append(int(remainder))
        return coefs

    def set_up_space_simple(self, space):
        """Given a space (percept or action space) of the type Discrete(single integer) or Box(tuple of integers) used in OpenAIGym,
        this function returns two flat lists and an integer:
            - cardinality runs over categories (ie independently discretized variables) and encodes how many discrete values the variable can take
            - discretization runs over simple spaces (each one is either Discrete or Box). For Discrete, it is None;
            for Box, it is a list of two arrays encoding offset and slope for translating discrete into continuous values.
            - space_type is 0 for Discrete and 1 for Box."""
        if isinstance(space, gym.spaces.Discrete):
            cardinality_list_local = [space.n]
            discretization_local = [None] # None - Discrete
            space_type = 0
        elif isinstance(space, gym.spaces.Box):
            #memo: make a global variable self.discretization_num_bins
            cardinality_list_local = space.low.size * [self.discretization_num_bins] #for Box(N1,N2,...), this yields a flat list of length N1*N2*... and all entries equal to discretization_num_bins
            discretization_local = [[space.low, (space.high-space.low)/self.discretization_num_bins]]
            space_type = 1
        return cardinality_list_local, discretization_local, space_type
    
    def set_up_space_generic(self, space):
        """Given a space (percept or action space) of the type Discrete, Box or Tuple,
        this function returns two flat lists and an integer:
            - cardinality runs over categories (ie independently discretized variables) and encodes how many discrete values the variable can take
            - discretization runs over simple spaces (each one is either Discrete or Box). For Discrete, it is None;
            for Box, it contains two arrays encoding offset and slope for translating discrete into continuous values.
            - space_type is 0 for a single Discrete, 1 for a single Box and 2 for Tuple."""
        if isinstance(space, gym.spaces.Discrete) or isinstance(space, gym.spaces.Box):
            cardinality_list, discretization_list, space_type = self.set_up_space_simple(space)
        elif isinstance(space, gym.spaces.Tuple):
            cardinality_list = []
            discretization_list = []
            for factor_space in space.spaces:
                cardinality_list_local, discretization_local, space_type = self.set_up_space_simple(factor_space)
                cardinality_list += cardinality_list_local
                discretization_list += discretization_local
            space_type = 2
        return cardinality_list, discretization_list, space_type    
        
    def observation_preprocess_simple(self, observation, discretize_observation): # preparing a discretized observation
        """Turns a raw observation from a space of type Discrete or Box 
        into a one-dimensional list of integers."""
        if type(observation) == np.ndarray:
            observation_discretized = ((observation - discretize_observation[0])/discretize_observation[1]).astype(int) #element-wise: (raw observation - offset) / slope; casting as integers automatically acts like floor function
            observation_discretized[observation_discretized >= self.discretization_num_bins] = self.discretization_num_bins - 1
            #safeguard against the case where the value of an observation is exactly at the upper bound of the range, which would give a discretized value outside the allowed range(self.discretization_num_bins)
            observation_preprocessed = list(observation_discretized.flatten())
        else:
            observation_preprocessed = [observation]
        return observation_preprocessed

    def observation_preprocess_generic(self, observation): # preparing a discretized observation
        """Turns a raw observation, which may be an array or tuple of arbitrary shapes containing continuous values, 
        into a one-dimensional list of integers."""
        if self.percept_space_type == 0 or self.percept_space_type == 1:
            observation_preprocessed = self.observation_preprocess_simple(observation, self.discretize_percepts_list[0])
        elif self.percept_space_type == 2:
            observation_preprocessed = []
            for tuple_index in range(len(self.env.observation_space.sample())):
                observation_preprocessed += self.observation_preprocess_simple(observation[tuple_index],self.discretize_percepts_list[tuple_index])
        return observation_preprocessed
               
    def action_postprocess_simple(self, action_flattened, discretize_action):
        """For a single simple space (Discrete or Box), given a flat list of integer action indices that runs over the categories in that space
        and the appropriate discretization information (None or [array_offset, array_slope], respectively),
        this method returns an integer (for Discrete) or an array of continuous variables (for Box)."""
        if discretize_action == None:
            action = action_flattened[0] #This takes just the integer, not a list
        else:
            action_reshaped = np.array(action_flattened).reshape(discretize_action[0].shape) #use offset array to get the right shape
            action = discretize_action[0] + (action_reshaped + 0.5) * discretize_action[1] #offset + (discrete+1/2)*slope
        return action
    
    def action_postprocess_generic(self, action_index):
        """Given a single integer action index, this function unpacks it back into an integer (for a single Discrete space), 
        an array of continuoues values (for a single Box space), or a tuple of several of those (in the case of a Tuple space)."""
        #decompose a single action index into indices for the different categories, whose cardinalities are given in self.num_actions_list
        action_flattened = self.decompose_variable_base(action_index, self.num_actions_list)
        if self.action_space_type == 0 or self.action_space_type == 1:
            action = self.action_postprocess_simple (action_flattened, self.discretize_actions_list[0])
        elif self.action_space_type == 2:
            category_index = 0 #runs over categories, viz elements of action_flattened
            action = () #This tuple will collect the actions (generally arrays of continuous variables) from all Discretes / Boxes
            for tuple_index in range(len(self.env.action_space.sample())):
                if self.discretize_actions_list[tuple_index] == None:
                    categories_in_subspace = 1
                else:
                    categories_in_subspace = len(self.discretize_actions_list[tuple_index][0])
                action_subset = action_flattened[category_index:category_index+categories_in_subspace]
                print(self.action_postprocess_simple(action_subset, self.discretize_actions_list[tuple_index]))
                action += tuple([self.action_postprocess_simple(action_subset, self.discretize_actions_list[tuple_index])])
                category_index += categories_in_subspace
        return action

## METHODS TO BE USED FROM OUTSIDE: INIT, RESET, MOVE

    def __init__(self, openai_env_name, discretization_num_bins=10):
        """Initialize an environment, specified by its name, given as a string. 
        A list of existing environments can be found at https://gym.openai.com/envs/;
        examples include 'CartPole-v1' and 'MountainCar-v0'.
        Optional argument: discretization_num_bins, for the case of continuous percept spaces."""
        self.env = gym.make(openai_env_name)
        self.discretization_num_bins = discretization_num_bins
        self.num_percepts_list, self.discretize_percepts_list, self.percept_space_type = self.set_up_space_generic(self.env.observation_space)
        self.num_actions_list, self.discretize_actions_list, self.action_space_type = self.set_up_space_generic(self.env.action_space)
        self.num_actions = np.prod(self.num_actions_list)

    def reset(self):
        """Reset environment and return (preprocessed) new percept."""
        observation = self.env.reset()
        return self.observation_preprocess_generic(observation)
		
    def move(self, action):
        """Given an action (single integer index), this method uses action_postprocess to put it in the format
        expected by the OpenAIGym environment, applies the action and returns the resulting new percept, reward and trial_finished.
        The percept is again preprocessed into a one-dimensional list of integers."""
        observation, reward, trial_finished, info = self.env.step(self.action_postprocess_generic(action))
        discretized_observation = self.observation_preprocess_generic(observation)
        return discretized_observation, reward, trial_finished
