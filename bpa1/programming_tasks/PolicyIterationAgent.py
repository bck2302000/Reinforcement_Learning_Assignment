import numpy as np
import random
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        random.seed(42)
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        # self.V = ...
        self.V = {s : 0 for s in states}

        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        counter = 0
        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    # if...
                    #
                    # else:...
                    newV[s] = 0
                    if s != self.mdp.terminalState:
                        # result of getTransitionStatesAndProbs will be like [[new_state_1, prob_1], [new_state_2, prob_2]]
                        ### For random policy
                        # for action in self.mdp.getPossibleActions(s):
                        #     for real_dir in self.mdp.getTransitionStatesAndProbs(s, action):
                        #         newV[s] += 0.25 * real_dir[1] * (reward + self.discount * self.V[real_dir[0]])
                        newV[s] = self.getQValue(s, a)
                
                self.V = newV



                # update value estimate
                # self.V=...

                # ******************

            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    # self.pi[s] = ...

                    # policy_stable =
                    reward = self.mdp.getReward(s, None, None)
                    pi_candidates = []
                    for action in actions:
                        q_val = self.getQValue(s, action)
                        pi_candidates.append(q_val)
                    self.pi[s] = actions[pi_candidates.index(max(pi_candidates))]
                    
                    if self.pi[s] != old_action:
                        policy_stable = False

                    # ****************
            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.

        return self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
        
        q_val = 0
        for real_dir in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val += real_dir[1] * (self.mdp.getReward(state, None, None) + self.discount * self.V[real_dir[0]])

        return q_val
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 1.4.

        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
