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
        self.visit_n = np.zeros(tuple(self._stateSpaceShape) + (self._numActions, ))

    def GetAction(self,
                  currentState,
                  learningMode,
                  actionProbabilityBase=1.8,
                  randomActionRate=0.01):

        rand = random.random()
        if rand < randomActionRate:
            action = random.choice([x for x in range(self._numActions)])
            self.visit_n[tuple(currentState)][action] += 1
            return action

        if learningMode:
            q_hats = self.q_table[tuple(currentState)]
            deno = sum([actionProbabilityBase**q_hat for q_hat in q_hats])
            prob = [actionProbabilityBase**q_hat / deno for q_hat in q_hats]
            if len(set(prob)) == 1:
                action = random.choice([x for x in range(len(prob))])
            else:
                action = np.argmax(prob)
            self.visit_n[tuple(currentState)][action] += 1
            return action

        action = np.argmax(self.q_table[tuple(currentState)])
        self.visit_n[tuple(currentState)][action] += 1
        return action

    def ObserveAction(self,
                      oldState,
                      action,
                      newState,
                      reward,
                      learningRateScale):

        alpha_n = 1. / (1 + learningRateScale * self.visit_n[tuple(oldState)][action])  # (13.11)
        updates = (1 - alpha_n) * self.q_table[tuple(oldState) + (action,)] + alpha_n * (reward + self._discountRate * np.max(self.q_table[tuple(newState)]))
        self.q_table[tuple(oldState) + (action,)] = updates
