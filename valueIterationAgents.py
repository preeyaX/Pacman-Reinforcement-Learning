# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        currentStates = self.mdp.getStates()
        for iter in range(self.iterations):

            currentValues = self.values.copy()
            for state in currentStates:
                
                if self.mdp.isTerminal(state):
                    continue

                actions = self.mdp.getPossibleActions(state)
                possibleValues = []

                for action in actions:
                    t = self.mdp.getTransitionStatesAndProbs(state, action)

                    summation = 0
                    for successorState, prob in t:
                        reward = self.mdp.getReward(state, action, successorState)
                        summation += prob * (reward + (self.discount * currentValues[successorState]))
                    possibleValues.append(summation)

                self.values[state] = max(possibleValues)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        t = self.mdp.getTransitionStatesAndProbs(state, action)

        summation = 0
        for successorState, prob in t:
            reward = self.mdp.getReward(state, action, successorState)
            summation += prob * (reward + (self.discount * self.values[successorState]))

        return summation

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)

        allQValues = []
        for action in actions:
            allQValues.append(self.computeQValueFromValues(state, action))

        return actions[allQValues.index(max(allQValues))]
    
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        currentStates = self.mdp.getStates()
        stateCounter = 0

        for iter in range(self.iterations):
            currentValues = self.values.copy()
            
            state = currentStates[stateCounter]
                
            if self.mdp.isTerminal(state):
                if stateCounter == len(currentStates) - 1:
                    stateCounter = 0
                else:
                    stateCounter += 1
                continue

            actions = self.mdp.getPossibleActions(state)
            possibleValues = []

            for action in actions:
                t = self.mdp.getTransitionStatesAndProbs(state, action)

                summation = 0
                for successorState, prob in t:
                    reward = self.mdp.getReward(state, action, successorState)
                    summation += prob * (reward + (self.discount * currentValues[successorState]))
                possibleValues.append(summation)
            
            if stateCounter == len(currentStates) - 1:
                stateCounter = 0
            else:
                stateCounter += 1

            self.values[state] = max(possibleValues)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Compute predecessors of all states.
        allStates = self.mdp.getStates()

        predecessors = []
        for i in range(len(allStates)):
            predecessors.append([])

        for state in allStates:
            actions = self.mdp.getPossibleActions(state)

            for action in actions:
                t = self.mdp.getTransitionStatesAndProbs(state, action)
                for successorState, prob in t:
                    if prob > 0:
                        predecessors[allStates.index(successorState)].append(state)

        # Initialize an empty priority queue.
        pq = util.PriorityQueue()
        
        # For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate over states in the order returned by self.mdp.getStates())
            # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
            # Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        
        for s in allStates:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)

                if len(actions) == 0:
                    maxQValue = 0
                else:
                    maxQValue = max(self.computeQValueFromValues(s, a) for a in actions)

                currValue = self.getValue(s)
                diff = abs(maxQValue - currValue)
                pq.push(s, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if pq.isEmpty():
                break
            # Pop a state s off the priority queue.
            s = pq.pop()
            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                value = max(self.computeQValueFromValues(s, a) for a in actions)
                self.values[s] = value

            # For each predecessor p of s, do:
            for p in predecessors[allStates.index(s)]:
                # Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
                actions = self.mdp.getPossibleActions(p)

                if len(actions) == 0:
                    maxQValue = 0
                else:
                    maxQValue = max(self.computeQValueFromValues(p, a) for a in actions)

                currValue = self.getValue(p)
                diff = abs(maxQValue - currValue)
                # If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority. As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                if diff > self.theta:
                    pq.push(p, -diff)
