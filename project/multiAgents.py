from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostsPosition = currentGameState.getGhostPositions()

    if abs(pacmanPosition[0]-ghostsPosition[0][0]) + abs(pacmanPosition[1]-ghostsPosition[0][1]) == 1:
        return -10000

    return currentGameState.getScore()


def getPossibleActions(gameState, player):
    legalAction = gameState.getLegalActions(player)
    if Directions.STOP in legalAction:
        legalAction.remove(Directions.STOP)
    return legalAction


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def minimax(self, depth, gameState, player):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        elif player == 0:
            return self.getMax(depth, gameState, player)
        else:
            if player == gameState.getNumAgents():
                player = 0
            if player == 0:
                depth += 1
            return self.getMin(depth, gameState, player)

    def getMax(self, depth, gameState, player):
        maxValue = float('-inf')
        legalActions = getPossibleActions(gameState,player)
        for action in legalActions:
            temp = self.minimax(depth, gameState.generateSuccessor(player, action), 1)
            if temp > maxValue:
                maxValue = temp
        return maxValue

    def getMin(self, depth, gameState, player):
        minValue = float('inf')
        legalActions = getPossibleActions(gameState , player)
        for action in legalActions:
            temp = self.minimax(depth, gameState.generateSuccessor(player, action), player + 1)
            if temp < minValue:
                minValue = temp
        return minValue

    def getAction(self, gameState: GameState):
        legalActions = getPossibleActions(gameState , 0)
        bestAction = []
        for action in legalActions:
            bestAction.append(self.minimax(0,gameState.generateSuccessor(0,action),0))
        choosen = np.argmax(bestAction)
        print("Action : "+str(legalActions[choosen]))
        return legalActions[choosen]
