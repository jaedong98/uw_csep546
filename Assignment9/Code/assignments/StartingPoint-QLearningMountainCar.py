import os
import gym
from Assignment4Support import draw_accuracies
from Assignment9.Code import report_path

env = gym.make('MountainCar-v0')

import random
import QLearning # your implementation goes here...
import Assignment7Support


def qlearning(discountRate=0.98,
              actionProbabilityBase=1.8,
              trainingIteration=20000,
              randomActionRate=0.01,  # Percent of time the next action selected by GetAction is totally random
              learningRateScale = 0.01,
              mountainCarBinsPerDimension=20):

    qlearner = QLearning.QLearning(stateSpaceShape=Assignment7Support.MountainCarStateSpaceShape(mountainCarBinsPerDimension),
                                   numActions=env.action_space.n,
                                   discountRate=discountRate)

    for trialNumber in range(trainingIteration):
        observation = env.reset()
        reward = 0
        for i in range(201):
            #env.render()

            currentState = Assignment7Support.MountainCarObservationToStateSpace(observation, mountainCarBinsPerDimension)
            action = qlearner.GetAction(currentState, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase)

            oldState = Assignment7Support.MountainCarObservationToStateSpace(observation)
            observation, reward, isDone, info = env.step(action)
            newState = Assignment7Support.MountainCarObservationToStateSpace(observation)

            # learning rate scale
            qlearner.ObserveAction(oldState, action, newState, reward, learningRateScale=learningRateScale)

            if isDone:
                if(trialNumber%1000) == 0:
                    print(trialNumber, i, reward)
                break

    ## Now do the best n runs I can
    #input("Enter to continue...")

    n = 10
    totalRewards = []
    for runNumber in range(n):
        observation = env.reset()
        totalReward = 0
        reward = 0
        for i in range(201):
            #renderDone = env.render()

            currentState = Assignment7Support.MountainCarObservationToStateSpace(observation)
            observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, learningMode=False))

            totalReward += reward

            if isDone:
                #renderDone = env.render()
                print(i, totalReward)
                totalRewards.append(totalReward)
                break

    print(totalRewards)
    print("Your score:", sum(totalRewards) / float(len(totalRewards)))
    env.close()
    avg_score = sum(totalRewards) / float(len(totalRewards))
    return avg_score


if __name__ == '__main__':

    for i in range(1):
        mountain_cart_md = os.path.join(report_path, 'mt_cart_{}.md'.format(i))

        discountRates = [0.5, 0.6, 0.7, 0.8, 0.9]  # Controls the discount rate for future rewards -- this is gamma from 13.10
        actionProbabilityBases = [0.5, 1.0, 1.5, 2.0, 2.5]  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
        trainingIterations = [1000, 5000, 10000, 15000, 20000]
        BinsPerDimensions = [10, 14, 18, 22, 26]

        # discount rates
        # discountRates_img_fname = os.path.join(report_path, r'img/mt_cart_discountRate.png')
        # data = []
        # for discountRate in discountRates:
        #     avg_score = qlearning(discountRate=discountRate)
        #     data.append((discountRate, avg_score))
        #
        # draw_accuracies([data],
        #                 'Discount Rates',
        #                 'Avg. Score',
        #                 'Param Sweep - Discount Rate',
        #                 discountRates_img_fname, [], title_y=1.03)

        # actionProbabilityBases
        # actionProbabilityBases_img_fname = os.path.join(report_path, r'img/mt_cart_actionProbabilityBases.png')
        # data = []
        # for actionProbabilityBase in actionProbabilityBases:
        #     avg_score = qlearning(actionProbabilityBase=actionProbabilityBase)
        #     data.append((actionProbabilityBase, avg_score))
        #
        # draw_accuracies([data],
        #                 'Action Probability Bases',
        #                 'Avg. Score',
        #                 'Param Sweep - Action Probability Bases',
        #                 actionProbabilityBases_img_fname, [], title_y=1.03)

        # trainingIterations
        # trainingIterations_img_fname = os.path.join(report_path, r'img/mt_cart_trainingIterations.png')
        # data = []
        # for trainingIteration in trainingIterations:
        #     avg_score = qlearning(trainingIteration=trainingIteration)
        #     data.append((trainingIteration, avg_score))
        #
        # draw_accuracies([data],
        #                 'Training Iterations',
        #                 'Avg. Score',
        #                 'Param Sweep - Training Iterations',
        #                 trainingIterations_img_fname, [], title_y=1.03)

        # BinsPerDimensions
        BinsPerDimensions_img_fname = os.path.join(report_path, r'img/mt_cart_BinsPerDimensions.png')
        data = []
        for BinsPerDimension in BinsPerDimensions:
            avg_score = qlearning(mountainCarBinsPerDimension=BinsPerDimension)
            data.append((BinsPerDimension, avg_score))

        draw_accuracies([data],
                        'Bins Per Dimensions',
                        'Avg. Score',
                        'Param Sweep - Bins Per Dimensions',
                        BinsPerDimensions_img_fname, [], title_y=1.03)