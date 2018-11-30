import numpy as np
import random
random.seed(1)

class QLearning(object):

    def __init__(self, stateSpaceShape, numActions, discountRate):
        self._stateSpaceShape = stateSpaceShape
        self._numActions = numActions
        self._discountRate = discountRate
        self.state = None
        self.action = None

        # create a Q Table
        self.q_table = np.zeros(tuple(self._stateSpaceShape) + (self._numActions, ))
        self.visit_n = np.zeros(self.q_table.shape)

    def GetAction(self,
                  currentState,
                  learningMode,
                  actionProbabilityBase=1.8,
                  randomActionRate=0.01):
        action = np.argmax(self.q_table[tuple(currentState)])
        return action
        if not learningMode:
            return action

    def ObserveAction(self,
                      oldState,
                      action,
                      newState,
                      reward,
                      learningRateScale):

        alpha_n = 1 / (1 + learningRateScale * self.visit_n[tuple(newState) + (action,)])  # (13.11)
        updates = (1 - alpha_n) * self.q_table[tuple(oldState) + (action,)] + alpha_n * (self._discountRate + np.argmax(self.q_table[tuple(newState)]))
        self.q_table[tuple(oldState) + (action,)] = updates
        self.visit_n[tuple(newState), (action,)] += 1
        #best_q = np.amax(self.q_table[oldState])
        #self.q_table[tuple(oldState) + (action,)] += learningRateScale * (reward + self._discountRate * (best_q) - self.q_table[tuple(newState) + (action,)])

