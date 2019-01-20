

import queue

import datetime

import resource

import sys

import math

import heapq

from enum import Enum

class ColType(Enum):
         LIST = 1
         QUEUE = 2
         HEAP = 3

class Frontier(object):
    def __init__(self, colType):
        if colType == ColType.LIST:
            self._ordered == list()
        elif colType == ColType.QUEUE:
            self._ordered = queue.Queue()
        else:
            self._ordered = []
            heapq.heapify(self._ordered)
        self._set = set()
        self.colType = colType

    def add(self, element):
        if self.colType == ColType.QUEUE:
            self._ordered.put(element)
        elif self.colType == ColType.LIST:
            self._ordered.append(element)
        else:
            heapq.heappush(self._ordered, element)
        self._set.add(element.config)

    def get(self):
        element = self._ordered.get() if self.colType == ColType.QUEUE else (self._ordered.pop() if self.colType == ColType.LIST else heapq.heappop(self._ordered))
        self._set.remove(element.config)
        return element

    def contains(self, element):
        return element.config in self._set

    def decreaseKey(self, element):
        if self.colType != ColType.HEAP:
            raise Exception('Wrong collection type')
        return heapq._siftdown(self._ordered, 0, list(map(lambda x: x.config, self._ordered)).index(element.config))

    def isEmpty(self):
        return not self._ordered

#### SKELETON CODE ####

## The Class that Represents the Puzzle

class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n*n != len(config) or n < 2:

            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []


        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = int(i / self.n)

                self.blank_col = i % self.n

                break

    def get_distance(self, p1, p2):
        rowP1 = int(p1 / self.n)
        colP1 = p1 % self.n
        rowP2 = int(p2 / self.n)
        colP2 = p2 % self.n
        return abs(rowP1 - rowP2) + abs(colP1 - colP2)


    def get_manhattan_distance_from_goal(self):
        retVal = 0;
        for p1 in range(0, self.n * self.n):
            p2 = self.config[p1]
            retVal += self.get_distance(p1, p2)
        return retVal

    def __lt__(self, other):
        return self.get_manhattan_distance_from_goal() < other.get_manhattan_distance_from_goal()

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print(line)

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:

                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:

                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:

                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:

                self.children.append(right_child)

        return self.children


    def get_adam(self):
        retVal = self
        while retVal.parent is not None:
            retVal = retVal.parent
        return retVal


    def get_max_depth(self, max = 0):
        retVal = max
        if self.cost > max:
            retVal = self.cost
        for child in self.children:
            childMax = child.get_max_depth(retVal)
            if childMax > max:
                retVal = childMax;
        return retVal

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters

def writeOutput(state, nodesExpanded, runTime, maxRam = None):
    path = []
    currentState = state
    move = ''
    while currentState.action != 'Initial':
        path.append(currentState.action)
        currentState = currentState.parent

    path.reverse()

    print(f"path_to_goal: {','.join(path)}")
    print(f"cost_of_path: {state.cost}")
    print(f"nodes_expanded: {nodesExpanded}")
    print(f"search_depth: {len(path)}")
    # adam = state.get_adam()
    # maxDepth = adam.get_max_depth()
    # print(f"max_depth: {maxDepth}")
    print(f"running_time: {runTime.total_seconds()}")
    mb = 1024 * 1024;
    print(f"max_ram_usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / mb}")

    ### Student Code Goes here

def bfs_search(initial_state):

    """BFS search"""

    frontier = Frontier(ColType.QUEUE)

    frontier.add(initial_state)
    explored = set()
    nodes_expanded = 0

    while not frontier.isEmpty():
        state = frontier.get()
        explored.add(state.config)
        if test_goal(state):
            return {"state": state, "expansions": nodes_expanded}
        state.expand()
        nodes_expanded += 1
        if nodes_expanded % 1000 == 0:
            print(f"nodes expanded: {nodes_expanded}")
        for child in state.children:
            if child.config not in explored and not frontier.contains(child):
                frontier.add(child)
    return None


def dfs_search(initial_state):
    frontier = Frontier(ColType.LIST)
    frontier.add(initial_state)
    explored = set()
    nodes_expanded = 0

    while not frontier.isEmpty():
        state = frontier.get()
        explored.add(state.config)
        if test_goal(state):
            return {"state": state, "expansions": nodes_expanded}
        state.expand()
        nodes_expanded += 1
        if nodes_expanded % 1000 == 0:
            print(f"nodes expanded: {nodes_expanded}")
        for child in state.children:
            if child.config not in explored and not frontier.contains(child):
                frontier.add(child)
    return None

def A_star_search(initial_state):
    frontier = Frontier(ColType.HEAP)
    frontier.add(initial_state)
    explored = set()
    nodes_expanded = 0

    while not frontier.isEmpty():
        state = frontier.get()
        explored.add(state.config)
        if test_goal(state):
            return {"state": state, "expansions": nodes_expanded}
        state.expand()
        nodes_expanded += 1
        if nodes_expanded % 1000 == 0:
            print(f"nodes expanded: {nodes_expanded}")
        for child in state.children:
            if child.config not in explored and not frontier.contains(child):
                frontier.add(child)
            elif frontier.contains(child):
                frontier.decreaseKey(child)
    return None

def calculate_total_cost(state):
    pass

    """calculate the total estimated cost of a state"""

    ### STUDENT CODE GOES HERE ###

def calculate_manhattan_dist(idx, value, n):
    pass

    """calculatet the manhattan distance of a tile"""

    ### STUDENT CODE GOES HERE ###

def test_goal(puzzle_state):

    """test the state is the goal state or not"""

    last = -1
    for i, item in enumerate(puzzle_state.config):
        if item - last != 1:
            return False
        last = item
    return True

# Main Function that reads in Input and Runs corresponding Algorithm

def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    start = datetime.datetime.now()

    if sm == "bfs":

        finalState = bfs_search(hard_state)

    elif sm == "dfs":

        finalState = bfs_search(hard_state)

    elif sm == "ast":

        finalState = A_star_search(hard_state)

    else:

        print("Enter valid command arguments !")

    end = datetime.datetime.now()

    if finalState is not None:
        writeOutput(finalState['state'], finalState['expansions'], end - start)
    else:
        print("Epic Fail")

if __name__ == '__main__':

    main()