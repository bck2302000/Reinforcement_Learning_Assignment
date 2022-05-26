from agent import Agent
import numpy as np

# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.2):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.

        if state in self.Q and len(self.Q[state]) > 0:
            q_val = self.Q[state].values()
            return max(q_val)
        else:
            return self.qInitValue

        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if state in self.Q and action in self.Q[state]:
            return self.Q[state][action]
        else:
            return self.qInitValue
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        if state in self.Q and len(self.Q[state]) > 0:
            return max(self.Q[state], key=self.Q[state].get)
        else:
            return self.getRandomAction(state)
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            rand = np.random.randint(len(all_actions))
            return all_actions[rand]
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        rand = np.random.rand()
        if rand > self.epsilon:
            return self.getPolicy(state)
        else:
            return self.getRandomAction(state)

        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.

        next_max_Q = self.getValue(nextState)
        cur_Q = self.getQValue(state, action)
        td_error = self.learningRate * (reward + self.discount*next_max_Q - cur_Q)
        
        if state not in self.Q:
            for action in self.actionFunction(state):
                self.Q[state] = {action : self.qInitValue}
        
        self.Q[state][action] = cur_Q + td_error
             
        # *********
