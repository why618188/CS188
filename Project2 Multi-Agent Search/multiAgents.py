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



from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()

        ghostDistance, foodDistance = 99999, 99999
        for ghostState in newGhostStates:
            ghostDistance = min(ghostDistance, util.manhattanDistance(ghostState.getPosition(), newPos))
        for foodPosition in foodList:
            foodDistance = min(foodDistance, util.manhattanDistance(newPos, foodPosition))
        if len(foodList) < len(currentGameState.getFood().asList()):
            foodDistance = 0.5

        if ghostDistance > 4:
            return 5 / foodDistance
        return ghostDistance / foodDistance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial Project1 Search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial Project1 Search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        scores = [self.getActionRecursively(gameState.getNextState(0, action), 1, agentNum, self.depth) for action in
                  legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def getActionRecursively(self, gameState, currentIndex, agentNum, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if currentIndex == agentNum:
            return self.getActionRecursively(gameState, 0, agentNum, depth - 1)

        utility = []
        for action in gameState.getLegalActions(currentIndex):
            utility.append(self.getActionRecursively(gameState.getNextState(currentIndex, action), currentIndex + 1,
                                                        agentNum, depth))
        if currentIndex == 0:
            return max(utility)
        return min(utility)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()
        return self.maxValue(gameState, agentNum, self.depth, -99999, 99999)[1]

    def maxValue(self, gameState, agentNum, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'STOP'

        value = -99999
        for action in gameState.getLegalActions(0):
            newValue = self.minValue(gameState.getNextState(0, action), 1, agentNum, depth, alpha, beta)[0]
            if newValue > value:
                value, actionReturn = newValue, action
            if value > beta:
                return value, action
            alpha = max(alpha, value)
        return value, actionReturn

    def minValue(self, gameState, currentIndex, agentNum, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'STOP'
        if currentIndex == agentNum:
            return self.maxValue(gameState, agentNum, depth - 1, alpha, beta)

        value = 99999
        for action in gameState.getLegalActions(currentIndex):
            newValue = self.minValue(gameState.getNextState(currentIndex, action), currentIndex + 1, agentNum,
                                             depth, alpha, beta)[0]
            if newValue < value:
                value, actionReturn = newValue, action
            if value < alpha:
                return value, action
            beta = min(beta, value)
        return value, actionReturn


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()
        return self.getActionRecursively(gameState, 0, agentNum, self.depth)[1]

    def getActionRecursively(self, gameState, currentIndex, agentNum, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), 'stop'
        if currentIndex == agentNum:
            return self.getActionRecursively(gameState, 0, agentNum, depth - 1)

        if currentIndex == 0:
            value = -99999
            for action in gameState.getLegalActions():
                newValue = self.getActionRecursively(gameState.getNextState(0, action), 1, agentNum, depth)[0]
                if newValue > value:
                    value = newValue
                    actionReturn = action
            return value, actionReturn

        actionList = gameState.getLegalActions(currentIndex)
        value = 0
        for action in actionList:
            value += self.getActionRecursively(gameState.getNextState(currentIndex, action), currentIndex + 1, agentNum,
                                               depth)[0] / len(actionList)
        return value, 'stop'


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()

    food = currentGameState.getFood()
    foodList = food.asList()
    foodLeft = len(foodList)
    foodDistance = 99999
    for foodPosition in foodList:
        foodDistance = min(foodDistance, util.manhattanDistance(position, foodPosition))
    if foodLeft == 0:
        foodDistance = 0
    foodValue = 10 / (foodDistance + 1) - 2 * foodLeft

    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    ghostValue = 0
    for index in range(len(scaredTimes)):
        distance = util.manhattanDistance(position, ghostPositions[index])
        if scaredTimes[index] != 0:
            ghostValue += 100 / ((distance + 1) ** 3) * (scaredTimes[index] ** 0.5)
        else:
            ghostValue -= 200 / ((distance + 1) ** 3)

    return currentGameState.getScore() + foodValue + ghostValue

# Abbreviation
better = betterEvaluationFunction
