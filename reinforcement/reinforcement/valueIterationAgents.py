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

            #me defining mdp variables
        self.state = mdp.getStartState()
        self.possActions = mdp.getPossibleActions(self.state)
        #self.reward = mdp.getReward() #Get the reward for the state, action, nextState transition

        #reward for going north from start state
        print("reward for going north from start state: ", mdp.getReward(self.state, action = "north", nextState = (0,1)))

        #reachable states and their probabilities for the start state
        self.reachableStatesAndProbs = []
        
        totalReachableStates = []
        for action in self.possActions:
            print("action ", action, sep = ": ", end = " ")
            self.reachableStatesAndProbs += mdp.getTransitionStatesAndProbs(self.state, action)
            print(", possible states and their probabilities (nextState, prob): ", mdp.getTransitionStatesAndProbs(self.state, action))
        print("reachable states and probs: ", self.reachableStatesAndProbs)
        

        #run value iteration using what we have
        self.runValueIteration()
        
        
        
    def runValueIteration(self):
        #print("iterations: ", self.iterations)
        #print("discount: ", self.discount)
        #print("values: ", self.values)
        #print("start state: ", self.state)
        #print("possible actions: ", self.possActions)

        
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for k in range(self.iterations):
            values = util.Counter()
            for state in states:
                if not self.mdp.isTerminal(state):
                    action = self.computeActionFromValues(state)
                    values[state] = self.computeQValueFromValues(state,action)
            self.values = values
    
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
              
            getTransitionStatesAndProbs:
            Returns list of (nextState, prob) pairs
            representing the states reachable
            from 'state' by taking 'action' along
            with their transition probabilities.
        
        "*** YOUR CODE HERE ***"

              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        qVal = 0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state,action):
            reward = self.mdp.getReward(state, action, nextState)
            discount = self.discount
            nextValue = self.values[nextState]
            print("nextValue: ", nextValue, self.values)
            qVal += probability*(reward+discount*nextValue)
        
        return qVal
    
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #for state in state.getLegalActions():
           # return state
        
        possibleActions = self.mdp.getPossibleActions(state)
        actions = util.Counter()
       
        if self.mdp.isTerminal(state) == True:
            return None
         
        for action in possibleActions:
            qVal = self.computeQValueFromValues(state,action)
            actions[action] = qVal
       
        return actions.argMax()
                
        util.raiseNotDefined()

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

