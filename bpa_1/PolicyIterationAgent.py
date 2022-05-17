import numpy as np
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
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        # self.V = ...

        self.V = {s: 0 for s in states}

        # *******************

        self.pi = {s: self.mdp.getPossibleActions(
            s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

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

                    if a == None:
                        newV[s] = 0.0
                    else:
                        # for deterministic policy
                        newV[s] = self.getQValue(s, a)

                # update value estimate
                # self.V=...

                self.V = newV

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

                    q_value = [self.getQValue(s, a) for a in actions]
                    action_idx = np.argmax(q_value)
                    self.pi[s] = actions[action_idx]

                    # policy_stable =

                    if self.pi[s] != old_action:
                        policy_stable = False

                    # ****************
            counter += 1

            if policy_stable:
                break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.

        # ********
        return self.V[state]

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

        # **********
        # get the reward for current (s,a)
        r = self.mdp.getReward(state, action, None)

        # list_pair is like [[next_state_1, prob1], [xx_2, prob2]]
        list_pair = self.mdp.getTransitionStatesAndProbs(state, action)

        # sum over next_state bcs r is independent of next_state
        q_value = sum(
            [p * (r + self.discount * self.V[next_state])
             for next_state, p in list_pair]
        )

        return q_value

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 1.4.

        # **********
        return self.pi[state]

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
