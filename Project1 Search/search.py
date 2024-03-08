# Project1 Search.py
# ---------
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


"""
In Project1 Search.py, you will implement generic Project1 Search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a Project1 Search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the Project1 Search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the Project1 Search tree first.

    Your Project1 Search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph Project1 Search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the Project1 Search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()
    frontier.push(problem.getStartState())
    expanded = []
    path = {problem.getStartState(): []}

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node):
            return path[node]
        if node not in expanded:
            expanded.append(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                frontier.push(successor)
                path_to_node = path[node][:]
                path_to_node.append(action)
                path[successor] = path_to_node

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the Project1 Search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    frontier.push(problem.getStartState())
    visited = [problem.getStartState()]
    path = {problem.getStartState(): []}

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node):
            return path[node]
        for successor, action, stepCost in problem.getSuccessors(node):
            if successor in visited:
                continue
            visited.append(successor)
            frontier.push(successor)
            path_to_node = path[node][:]
            path_to_node.append(action)
            path[successor] = path_to_node

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"  # This function is not written by myself.
    visited = {}
    solution = []
    queue = util.PriorityQueue()
    route = {}
    cost = {}

    start = problem.getStartState()
    queue.push((start, '', 0), 0)
    visited[start] = ''
    cost[start] = 0

    if problem.isGoalState(start):
        return solution

    flag = False
    while not (queue.isEmpty() or flag):
        vertex = queue.pop()
        visited[vertex[0]] = vertex[1]
        if problem.isGoalState(vertex[0]):
            child = vertex[0]
            flag = True
            break
        for i in problem.getSuccessors(vertex[0]):
            if i[0] not in visited.keys():
                priority = vertex[2] + i[2]
                if not(i[0] in cost.keys() and cost[i[0]] <= priority):
                    queue.push((i[0], i[1], vertex[2] + i[2]), priority)
                    cost[i[0]] = priority
                    route[i[0]] = vertex[0]

    while(child in route.keys()):
        parent = route[child]
        solution.insert(0, visited[child])
        child = parent

    return solution

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    frontier.update(problem.getStartState(), heuristic(problem.getStartState(), problem))
    close = []
    path = {problem.getStartState(): []}
    cost = {problem.getStartState(): 0}
    cost_in_total = {problem.getStartState(): heuristic(problem.getStartState(), problem)}

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node):
            return path[node]
        if node not in close:
            close.append(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                current_cost_in_total = cost[node] + stepCost + heuristic(successor, problem)
                if successor in cost_in_total and cost_in_total[successor] <= current_cost_in_total:
                    continue
                frontier.update(successor, current_cost_in_total)
                cost_in_total[successor] = current_cost_in_total
                cost[successor] = cost[node] + stepCost
                path_to_node = path[node][:]
                path_to_node.append(action)
                path[successor] = path_to_node

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
