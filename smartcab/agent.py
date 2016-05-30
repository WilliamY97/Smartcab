import random
from math import log
import matplotlib.pyplot as plt
from numpy import mean
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color

        self.color = 'red'  # override color

        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

###############################################################################################################

        # TODO: Initialize any additional variables here

        #states
        recomendation_states = ['forward', 'left', 'right']
        light_states = ['green', 'red']
        actions_states = [None, 'forward', 'left', 'right']
        Q_key_table = []

        for action in actions_states:
            for left in actions_states:
                for oncoming in actions_states:
                    for light in light_states:
                        for recomend_nwp in recomendation_states:
                            Q_key_table.append((recomend_nwp,light,oncoming,left,action))

        Q_init = [random.random() * 4 for _ in range(0, len(Q_key_table))]

        self.Q_table = dict(zip(Q_key_table,Q_init))

        #rewards configuration
        self.totalR = 0.0         #total reward    
        self.num_actions = 1
        self.totalR_list = []
        self.average_reward_list = []

###############################################################################################################

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # tracks performance of the cabs
        self.average_reward_list.append (mean(self.totalR_list))

        self.totalR_list.append(self.totalR/self.num_actions)

        # Chart
        plt.figure(1)
       
        plt.subplot(211)        
        plt.plot(self.totalR_list)
 
        plt.title('Learning Performance')

        plt.xlabel('Trails')
        plt.ylabel('Rewards & Steps')
        
        plt.subplot(212)
        plt.plot(self.average_reward_list)     
        
        plt.xlabel('Trails')
        plt.ylabel('Avrg Reward')  

        # Resets rewards
        self.num_actions = 1

        self.totalR = 0.0  




    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)



        # TODO: Update state


        
        state = ((self.next_waypoint,) + (inputs['light'], inputs['oncoming'], inputs['left']))
        



        self.state = "nwp: {} / lights: {} / oncoming: {} / rows: {}".format(state[0],state[1],state[2],state[3])

 

        # TODO: Select action according to your policy




        #Grab the Q-values of possible cations and put them into the decision table
        decisiontable = {None: self.Q_table[(state + (None,))], 'right': self.Q_table[(state + ('right',))], 'left': self.Q_table[(state + ('left',))], 'forward': self.Q_table[(state + ('forward',))]}

        # Rate of exploration
        rate_of_exploration = (log(deadline+0.0001)) * 0.0155

        if  rate_of_exploration < random.random():

            # chooses the Q-value and action pair that provides the greatest Q-value
            qval_present, action = max((v, k) for k, v in decisiontable.iteritems())

        else:
            # chooses a random Q-value and action pair with the greatest Q-value
            action =  random.choice([None, 'forward', 'left', 'right'])
            qval_present = decisiontable [(action)]




        # Gets reward for performance
        reward = self.env.act(self, action)




        # TODO: Learn policy based on state, action, reward




          # Initializes the learning rate to prepare for policy
        learning_rate = (1.0 / (t+5)) + 0.75
        
        discount = 0.4 #40%

        newinputs = self.env.sense(self)
        newstate = ((self.next_waypoint,) + (newinputs['light'], newinputs['oncoming'], newinputs['left']))


        future_table = {None: self.Q_table[(newstate + (None,))], 'right': self.Q_table[(newstate + ('right',))], 'left': self.Q_table[(newstate + ('left',))], 'forward': self.Q_table[(newstate + ('forward',))]}

        future_reward, future_action = max((v, k) for k, v in future_table.iteritems())

        
        new_qval = reward + discount * future_reward



        self.Q_table[(state + (action,))] = qval_present + learning_rate * (new_qval - qval_present)
        self.num_actions = self.num_actions + 1   
        self.totalR = self.totalR + reward

       

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=30)  # press Esc or close pygame window to quit
    
    plt.show()
   

if __name__ == '__main__':
    run()

