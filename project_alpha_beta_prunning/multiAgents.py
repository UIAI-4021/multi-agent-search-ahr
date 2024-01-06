# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

import time

def EvaluationFunction(currentGameState: GameState):

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Consts
    INF = 100000000.0  # Infinite value for being dead
    WEIGHT_FOOD = 5.0  # Food base value
    WEIGHT_GHOST = -10.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 10.0  # Scared ghost base value

    # Base on gameState.getScore()

    score = currentGameState.getScore()

    # Evaluate the distance to the closest food
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(distancesToFoodList) > 0:
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        score += WEIGHT_FOOD

    # Evaluate the distance to ghosts
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # If scared, add points
                score += WEIGHT_SCARED_GHOST / distance
            else:  # If not, decrease points
                score += WEIGHT_GHOST / distance
        else:
            return -INF  # Pacman is dead at this point
    #print(score)
    return score


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        #self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):


        curValue, alpha, beta = -1e9, -1e9, 1e9
        nextPacmanAction = Directions.STOP
        #print(gameState.getLegalActions(0))
        legalActions = gameState.getLegalActions(0)
        #print(legalActions)

        for nextAction in legalActions:
            nextState = gameState.generateSuccessor(0, nextAction)

            nextValue = self.getNodeValue(nextState, 0, 1, alpha, beta)

            if nextValue > curValue:
                curValue, nextPacmanAction = nextValue, nextAction

            alpha = max(alpha, curValue)

        #print(f"pacmna is at {gameState.getPacmanPosition()} the action is : {nextPacmanAction} the value is {curValue}")
        return nextPacmanAction

    def getNodeValue(self, gameState, depth=0, agentIdx=0, alpha=-1e9, beta=1e9):

        maxParty = [0, ]
        minParty = list(range(1, gameState.getNumAgents()))

        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return EvaluationFunction(gameState)
        elif agentIdx in maxParty:
            return self.alphaValue(gameState, depth, agentIdx, alpha, beta)
        elif agentIdx in minParty:
            return self.betaValue(gameState, depth, agentIdx, alpha, beta)

    def alphaValue(self, gameState, depth, agentIdx, alpha=-1e9, beta=1e9):

        value = -1e9
        legalActions = gameState.getLegalActions(agentIdx)
        for index, action in enumerate(legalActions):
            nextValue = self.getNodeValue(gameState.generateSuccessor(agentIdx, action), \
                                          depth, agentIdx + 1, alpha, beta)
            value = max(value, nextValue)
            if value > beta:  # next_agent in which party
                return value
            alpha = max(alpha, value)
        return value

    def betaValue(self, gameState, depth, agentIdx, alpha=-1e9, beta=1e9):

        value = 1e9
        legalActions = gameState.getLegalActions(agentIdx)
        for index, action in enumerate(legalActions):
            if agentIdx == gameState.getNumAgents() - 1:
                nextValue = self.getNodeValue(gameState.generateSuccessor(agentIdx, action), \
                                              depth + 1, 0, alpha, beta)
                value = min(value, nextValue)  # begin next depth
                if value < alpha:
                    return value
            else:
                nextValue = self.getNodeValue(gameState.generateSuccessor(agentIdx, action), \
                                              depth, agentIdx + 1, alpha, beta)
                value = min(value, nextValue)  # begin next depth
                if value < alpha:  # next agent goes on at the same depth
                    return value
            beta = min(beta, value)
        return value