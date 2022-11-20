import numpy as np
import math
import heapq
from collections import namedtuple

from path_planning import generateMap2d

# Declaring namedtuple()
node = namedtuple('node', ['value','parent', 'location'])

# Priority Queue based on heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []
    def isEmpty(self):
        return len(self.elements) == 0
    def add(self, item):
        heapq.heappush(self.elements,(priority,item))
    def pop(self):
        return heapq.heappop(self.elements)[1]


# An example of search algorithm, feel free to modify and implement the missing part
def search(obs_map):

    map_shape = obs_map.shape

    # cost moving to another cell
    moving_cost = 1

    # open list
    frontier = [] # PriorityQueue()
    # add starting cell to open list

    # path taken
    came_from = {}

    # expanded list with cost value for each cell
    cost = {}

    # init. starting node
    start_location = None
    for y, c in enumerate(obs_map):
        if -2 in c:
            start_location = (list(c).index(-2), y)

    current_node = node(value=-2, parent=None, location=start_location)

    print("starting loc", start_location)

    frontier.append(current_node)
    

    while current_node.value != -3:
        current_node = frontier.pop(0)
        current_location = current_node.location

        # Next Nodes
        for l in [(0,1), (0,-1), (1,0), (-1,0)]:

            n_x = current_location[0] + l[0]
            n_y = current_location[1] + l[1]

            print(n_x,n_y)

            if n_x < map_shape[1] and n_x >= 0 \
                and n_y < map_shape[0] and n_y >= 0: 

                n = node(parent=current_node,
                         location=(n_x, n_y),
                         value=obs_map[n_x][n_y])
                
                if n.value == -1:   
                    pass
                elif n in frontier:
                    print(24354656554)
                    quit()
                else:
                    frontier.append(n)    
    
    print("found goal node", current_node.location)

"""
    # if there is still nodes to open
    while not frontier.isEmpty():
        current = frontier.remove()

        # check if the goal is reached
        if current == goal:
            break

        # for each neighbour of the current cell
        # Implement get_neighbors function (return nodes to expand next)
        # (make sure you avoid repetitions!)
        for next in get_neighbors(current):

            # compute cost to reach next cell
            # Implement cost function
            cost = cost_function()

            # add next cell to open list
            frontier.add(next, cost)
            
            # add to path
            came_from[next] = current
    return came_from, cost
"""

if __name__=="__main__":
    obs_map = generateMap2d([10,10])
    print(obs_map)
    path = search(obs_map)
    print(path)
