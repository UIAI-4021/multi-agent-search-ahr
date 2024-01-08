from game import Directions
import random, util
import numpy as np

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Consts
    INF = 100000000.0  # Infinite value for being dead
    WEIGHT_FOOD = 5.0  # Food base value
    WEIGHT_GHOST = -5.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 50.0  # Scared ghost base value

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
        distance = util.manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # If scared, add points
                score += WEIGHT_SCARED_GHOST / distance
            else:  # If not, decrease points
                score += WEIGHT_GHOST / distance
        else:
            return -INF  # Pacman is dead at this point
    # print(score)
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
    def alphaBeta(self, depth, gameState, player, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if player == 0:
            return self.alphaPart(depth, gameState, player, alpha, beta)
        else:
            return self.betaPart(depth, gameState, player, alpha, beta)

    def alphaPart(self, depth, gameState, player, alpha, beta):
        maxValue = float('-inf')
        legalActions = getPossibleActions(gameState, player)
        for action in legalActions:
            maxValue = max(maxValue, self.alphaBeta(depth, gameState.generateSuccessor(player, action), 1, alpha, beta))
            if maxValue >= beta:
                break
            alpha = max(alpha, maxValue)
        return maxValue

    def betaPart(self, depth, gameState, player, alpha, beta):
        nextPlayer = player + 1
        if player == gameState.getNumAgents() - 1:
            nextPlayer = 0
        if nextPlayer == 0:
            depth += 1

        minValue = float('inf')
        legalActions = getPossibleActions(gameState, player)
        for action in legalActions:
            minValue = min(minValue,
                           self.alphaBeta(depth, gameState.generateSuccessor(player, action), nextPlayer, alpha, beta))
            if minValue <= alpha:
                break
            beta = min(beta, minValue)
        return minValue

    def getAction(self, gameState: GameState):
        alpha = float('-inf')
        beta = float('inf')
        legalActions = getPossibleActions(gameState, 0)
        bestAction = []
        for action in legalActions:
            bestAction.append(self.alphaBeta(0, gameState.generateSuccessor(0, action), 1, alpha, beta))
        choosen = np.argmax(bestAction)
        max_indices = [index for index in range(len(bestAction)) if bestAction[index] == bestAction[choosen]]
        chosenIndex = random.choice(max_indices)
        return legalActions[chosenIndex]


class MiniMaxAgent(MultiAgentSearchAgent):
    def minimax(self, depth, gameState, player):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
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
            bestAction.append(self.minimax(0, gameState.generateSuccessor(0, action), 1))
        choosen = np.argmax(bestAction)
        max_indices = [index for index in range(len(bestAction)) if bestAction[index] == bestAction[choosen]]
        chosenIndex = random.choice(max_indices)
        return legalActions[chosenIndex]
