from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


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
        legalActions = gameState.getLegalActions(player)
        for action in legalActions:
            temp = self.minimax(depth, gameState.generateSuccessor(player, action), 1)
            if temp > maxValue:
                maxValue = temp
        return maxValue

    def getMin(self, depth, gameState, player):
        minValue = float('inf')
        legalActions = gameState.getLegalActions(player)
        for action in legalActions:
            temp = self.minimax(depth, gameState.generateSuccessor(player, action), player + 1)
            if temp < minValue:
                minValue = temp
        return minValue

    def getAction(self, gameState: GameState):
        legalActions = gameState.getLegalActions(0)
        bestAction = []
        for action in legalActions:
            bestAction.append(self.minimax(0,gameState.generateSuccessor(0,action),0))
        choosen = np.argmax(bestAction)
        print("Action : "+str(legalActions[choosen]))
        return legalActions[choosen]
