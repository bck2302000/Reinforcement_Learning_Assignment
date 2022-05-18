import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states

        # set Q is a dict mapping from state to action(dict)
        # action is a dict mapping action to q_value
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

        # greedy policy
        if state in self.Q and len(self.Q[state]) > 0:
            q_value = self.Q[state].values()
            return max(q_value)
        else:
            return self.qInitValue

        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.

        if state in self.Q and action in self.Q[state]:
            action_value_pair = self.Q[state]
            return action_value_pair[action]
        else:
            return self.qInitValue

        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        if state in self.Q and len(self.Q[state]) > 0:
            action_value_pair = self.Q[state]
            max_value_action = max(
                action_value_pair, key=action_value_pair.get
            )  # str action
            return max_value_action
        else:
            return self.getRandomAction(state)

        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            action_idx = np.random.randint(len(all_actions))
            return all_actions[action_idx]
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        rand_number = np.random.rand()
        if rand_number > self.epsilon:
            return self.getPolicy(state)
        else:
            return self.getRandomAction(state)

        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.


        curr_q_value = self.getQValue(state, action)  # float
        q_value_next_state = self.getValue(nextState) # max q
        td_error = (
            self.learningRate * (reward + self.discount *
                                 q_value_next_state - curr_q_value)
        )

        if state not in self.Q:
            all_actions = self.actionFunction(state)
            self.Q[state] = {a : self.qInitValue for a in all_actions}

        self.Q[state][action] = curr_q_value + td_error

        # *********
