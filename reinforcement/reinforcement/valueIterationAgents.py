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
<<<<<<< HEAD
        self.values = util.Counter() # A Counter is a dict with default 0

            #me defining mdp variables
=======
        self.values = util.Counter() #A Counter is a dict with default 0
        
        #me defining mdp variables
>>>>>>> 9b071118ae36ee3e0ea351e4c5b57395b8b42496
        self.state = mdp.getStartState()
        self.possActions = mdp.getPossibleActions(self.state)
        #self.reward = mdp.getReward() #Get the reward for the state, action, nextState transition

        
        #V(s) = max_{a in actions} Q(s,a)
        if self.mdp.getPossibleActions(self.state):
            possActions = self.mdp.getPossibleActions(self.state)
            qVals = util.Counter()
            for action in possActions:
                qVal = self.computeQValueFromValues(self.state,action)
                print("action, qVal, state:  ",action,qVal,self.state)
                qVals[action] = qVal
            #I think this sorts them...
            sortedQVals = [(k, qVals[k]) for k in sorted(qVals, key=qVals.get, reverse=True)]
            for k, v in sortedQVals:
                print(k, v)
            #self.values[action] = qVal
            #sortedVals = self.values.sortedKeys()


        #reachable states and their probabilities for the start state
        self.reachableStatesAndProbs = []
        
        for action in self.possActions:
            print("action ", action, sep = ": ", end = " ")
            self.reachableStatesAndProbs += mdp.getTransitionStatesAndProbs(self.state, action)
            print(", possible states and their probabilities (nextState, prob): ", mdp.getTransitionStatesAndProbs(self.state, action))
            print("values: ", self.values)
        print("reachable states and probs: ", self.reachableStatesAndProbs)
        

        #run value iteration using what we have
        self.runValueIteration()
<<<<<<< HEAD
        
        
        
=======


>>>>>>> 9b071118ae36ee3e0ea351e4c5b57395b8b42496
    def runValueIteration(self):
        #print("iterations: ", self.iterations)
        #print("discount: ", self.discount)
        #print("values: ", self.values)
        #print("start state: ", self.state)
        #print("possible actions: ", self.possActions)

        
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
<<<<<<< HEAD
        states = self.mdp.getStates()
        for k in range(self.iterations):
            values = util.Counter()
            for state in states:
                if not self.mdp.isTerminal(state):
                    action = self.computeActionFromValues(state)
                    values[state] = self.computeQValueFromValues(state,action)
            self.values = values
    
=======
        print("iterations: ", self.iterations)
        print("discount: ", self.discount)
        print("values: ", self.values)
        print("start state: ", self.state)
        print("possible actions: ", self.possActions)


        
        #reward = mdp.getReward(state, action, nextState)
        """
        value = 0
        if self.mdp.isTerminal(self.state):
            return 0
        self.values = {}
        for action in self.mdp.getPossibleActions(self.state):
            self.values[self.state] = 0
            for nextState,probability in self.mdp.getTransitionStatesAndProbs(self.state, action):
                reward = self.mdp.getReward(self.state, action, nextState)
                self.values[nextState] += probability*(reward + self.discount*1.1) #self.previous_values[nextState]
        return 0
        """

        #TERMINAL STATES ARE 0 VALUE

        #this was a stack overflow answer of valueiteration
        #actionCost[action] += probability * reward + discount * self.previous_values[nextState]
            
>>>>>>> 9b071118ae36ee3e0ea351e4c5b57395b8b42496
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
          V(s) = max_{a in actions} Q(s,a)
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
<<<<<<< HEAD
            reward = self.mdp.getReward(state, action, nextState)
            discount = self.discount
            nextValue = self.values[nextState]
            print("nextValue: ", nextValue, self.values)
            qVal += probability*(reward+discount*nextValue)
        
        return qVal
    
=======
            reward = self.mdp.getReward(self.state,action,nextState)
            qVal += probability * (reward + self.discount*self.getValue(state))
        return qVal
        
>>>>>>> 9b071118ae36ee3e0ea351e4c5b57395b8b42496
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
<<<<<<< HEAD
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
                
=======
        #self.values = util.Counter() #A Counter is a dict with default 0
        #Make sure there are legal actions
      

>>>>>>> 9b071118ae36ee3e0ea351e4c5b57395b8b42496
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

