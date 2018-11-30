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
                  randomActionRate,
                  actionProbabilityBase):
        return np.argmax(self.q_table[tuple(currentState)])

    def ObserveAction(self,
                      oldState,
                      action,
                      newState,
                      reward,
                      learningRateScale):
        best_q = np.amax(self.q_table[oldState])
        self.q_table[tuple(oldState) + (action,)] += learningRateScale * (reward + self._discountRate * (best_q) - self.q_table[tuple(newState) + (action,)])

