# Importing libraries

import numpy as np
import math
import random

# Defining the MDP (Markov Decision Process) hyperparameters
m = 5     # Number of cities, ranges from 0 ..... m-1
t = 24    # Number of hours, ranges from 0 .... t-1
d = 7     # Number of days, ranges from 0 ... d-1
C = 5     # Per hour fuel and other costs
R = 9     # Per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """Initialise the state and Define your action space and state space"""
        # All pairs of m where pickup!=drop except (0,0) No Action Taken
        self.action_space = [(pickup,drop) for pickup in range(m) for drop in range(m) if pickup!=drop or pickup==0]
        
        # All triplet combinations of m, t and d 
        self.state_space = [(curr_loc,time_hour,week_day) for curr_loc in range(m) for time_hour in range(t) for week_day in range(d)]
        
        # Random State from state_space
        self.state_init = random.sample(self.state_space,1)[0]
        
        self.total_time = 0                           # Total Episode Time (Hours)
        self.max_time = 24*30                         # Max Episode Time (Number of Hours in 1 month)
        self.action_size = m * (m-1) + 1              # Action Space Size
        self.state_size_arch_1 = m + t + d + m + m    # Architecture 1  State Size ( State + Action )
        self.state_size_arch_2 = m + t + d            # Architecture 2  State Size ( State )

        # Start the first round
        self.reset()
        


    ## Encoding state for NN input for Arch-Type-1
    
    def state_encod_arch1(self, state, action):
        """Convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. The vector is of size m + t + d + m + m."""
        state_encod = np.zeros(m + t + d + m + m)    # Initialise Empty State
        state_encod.reshape(1,m+t+d+m+m)             # Reshape 1 X State_size
        state_encod[state[0]] = 1                    # 1 in State Location
        state_encod[m+state[1]] = 1                  # 1 in State Hour 
        state_encod[m+t+state[2]] = 1                # 1 in State Day
        state_encod[m+t+m+action[0]] = 1             # 1 in Action Pick Location
        state_encod[m+t+m+m+action[1]] = 1           # 1 in Action Drop Location
        return state_encod

    ## Encoding state for NN input for Arch-Type-2

    def state_encod_arch_2(self, state):
        """Convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. The vector is of size m + t + d."""
        
        state_encod = np.zeros((m + t + d)) # Initialise Empty State
        state_encod[state[0]] = 1           # 1 in State Location
        state_encod[m + state[1]] = 1       # 1 in State Hour 
        state_encod[m+t+state[2]] = 1       # 1 in State Day

        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. Using Poisson Distribution"""
        
        location = state[0] # Current Location
        
        #Based on Current Location and Poisson Distribution get a random number        
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        # Max Requests = 15
        if requests > 15:
            requests = 15
        
        possible_actions_index = random.sample(range(1, self.action_size), requests) # Sample "requests" Number of Indexes from Action Space
        actions = [self.action_space[i] for i in possible_actions_index]  # Create list of possible Actions using given indexes
        
        actions.append((0,0))  # Appened 0,0 as No Request Accepted Action
        possible_actions_index.append(0)  # Appened 0 as No Request Accepted Action Index

        return possible_actions_index,actions



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        location, time_hour, week_day = state
        pickup,drop = action 
        
        reward = 0
        
        # If No Request Accepted Action (0,0)
        if pickup == 0 and drop == 0:
            reward = -C                # Reward = Cost for 1 hr
        else:
            # Initialise time taken from current location to pickup location
            pickup_time = 0
            
            # If current location and pickup location not the same, Calculate Pickup Location
            if location != pickup:
                pickup_time = Time_matrix[location,pickup,time_hour,week_day] # Time taken to Pickup Location based on current state and pickup location
                time_hour = int((time_hour + pickup_time)%t)                  # Calculate New Hour after pickup
                week_day = int((week_day + (time_hour + pickup_time)//t)%d)   # Calculate New Day after pickup
                
            drop_time = Time_matrix[pickup,drop,time_hour,week_day] # Time taken to Drop Location from pickup location
            
            reward = R*(drop_time) - C * ( pickup_time + drop_time ) # Reward = Revenue - Cost
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        location, time_hour, week_day = state
        pickup, drop = action 
        
        # If No Request Accepted Action (0,0)
        if pickup == 0 and drop == 0:
            self.total_time +=1                                   # Increase Total Time by 1 hr
            time_next = int(( time_hour + 1) % t)                 # Calculate New Hour after 1 hr
            next_day = int(( week_day + ( time_hour + 1)//t)%d)   # Calculate New Day after pickup
            next_location = location                              # Set Current Location as Next location
        else:
            trip_time = 0
            # If current location and pickup location not the same, Calculate Pickup Location
            if location != pickup:
                pickup_time = Time_matrix[location,pickup,time_hour,week_day]   # Time taken to Pickup Location based on current state and pickup location
                time_hour = int((time_hour + pickup_time)%t)                    # Calculate New Hour after pickup
                week_day = int((week_day + (time_hour + pickup_time)//t)%d)     # Calculate New Day after pickup
                trip_time+=pickup_time                                          # Add Pickup Time to Trip Time
                
                
            drop_time = Time_matrix[pickup,drop,time_hour,week_day]    # Time taken to Drop Location from pickup location
            time_next = int((time_hour + drop_time)%t)                 # Calculate New Hour after Drop
            next_day = int((week_day + (time_hour + drop_time)//t)%d)  # Calculate New Day after Drop
            trip_time+=drop_time                                       # Add Drop Time to Trip Time
            
            next_location = drop  # Set Drop Location as Next Location
            
            self.total_time += trip_time  # Add Trip time to Total Time
            
        next_state = (next_location,time_next,next_day)  # Create Next State
        
            
        is_terminal = self.total_time > self.max_time  # Check if Total Time greater than Max time, then Episode ends and is_terminal is True
        
        if is_terminal:  # If Is Terminal, Reset Env
            self.reset()
            
        return next_state,is_terminal

    # Resets the Env, Makes Total_Time as 0, returns a Initial State
    def reset(self):  
        self.total_time = 0
        return self.action_space, self.state_space, self.state_init
