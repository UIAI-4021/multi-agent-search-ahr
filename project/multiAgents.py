from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent
from pacman import GameState


def calculateDiff(position1, position2):
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])


def scoreEvaluationFunction(currentGameState: GameState):
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostsPosition = currentGameState.getGhostPositions()
    foods = currentGameState.getFood()
    foodMinDistance = 0
    foodDistances = [calculateDiff(pacmanPosition, foodPos) for foodPos in foods.asList()]
    if len(foodDistances) > 0:
        foodMinDistance = min(foodDistances)

    ghostsState = currentGameState.getGhostStates()
    scaredTimer = [ghost.scaredTimer for ghost in ghostsState]
    ghostsDistance = [calculateDiff(pacmanPosition, ghostPos) for ghostPos in ghostsPosition]

    ghostMinDistance = 0
    minScaredTime = 0

    if len(ghostsDistance) > 0:
        minGhostIndex = np.argmin(ghostsDistance)
        ghostMinDistance = ghostsDistance[minGhostIndex]
        minScaredTime = scaredTimer[minGhostIndex]

        if ghostMinDistance <= 1 and minScaredTime == 0:
            return -1000000
        elif ghostMinDistance <= 1 and minScaredTime > 0:
            return 1000000

    score = currentGameState.getScore() - foodMinDistance

    if minScaredTime > 0:
        score -= ghostMinDistance
    else:
        score += ghostMinDistance

    return score


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
            x = self.evaluationFunction(gameState)
            # print(x)
            # input()
            return x
        if player == 0:
            return self.getMax(depth, gameState, player)
        else:
            return self.getMin(depth, gameState, player)

    def getMax(self, depth, gameState, player):
        maxValue = float('-inf')
        legalActions = getPossibleActions(gameState, player)
        for action in legalActions:
            temp = self.minimax(depth, gameState.generateSuccessor(player, action), 1)
            if temp > maxValue:
                maxValue = temp
        return maxValue

    def getMin(self, depth, gameState, player):
        nextPlayer = player + 1
        if player == gameState.getNumAgents() - 1:
            nextPlayer = 0
        if nextPlayer == 0:
            depth += 1
        minValue = float('inf')
        legalActions = getPossibleActions(gameState, player)
        for action in legalActions:
            temp = self.minimax(depth, gameState.generateSuccessor(player, action), nextPlayer)
            if temp < minValue:
                minValue = temp
        return minValue

    def getAction(self, gameState: GameState):
        legalActions = getPossibleActions(gameState, 0)
        bestAction = []
        for action in legalActions:
            bestAction.append(self.minimax(0, gameState.generateSuccessor(0, action), 0))
        choosen = np.argmax(bestAction)
        print("Action : " + str(legalActions[choosen]))
        max_indices = [index for index in range(len(bestAction)) if bestAction[index] == bestAction[choosen]]
        chosenIndex = random.choice(max_indices)
        return legalActions[chosenIndex]
