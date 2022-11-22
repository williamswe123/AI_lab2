# from search_algorithm import
from path_planning import generateMap2d, generateMap2d_obstacle, plotMap
import random
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import namedtuple

# Declaring namedtuple()
node = namedtuple('node', ['value', 'parent', 'location'])

# Priority Queue based on heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def isEmpty(self):
        return len(self.elements) == 0

    def add(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def remove(self):
        return heapq.heappop(self.elements)[1]



def start(obs_map):
    start_location = node(value=None, parent=None, location=None)
    for x, c in enumerate(obs_map):
        if -2 in c:
            start_location = node(value=-2, parent=None, location=(x, list(c).index(-2)))
    return start_location


def get_neighbors(obs_map, current_node):
    map_shape = obs_map.shape
    neighbors = []
    current_location = current_node.location
    # Next Nodes
    for l in [(0, 1), (0, -1), (1, 0), (-1, 0)]:

        n_x = current_location[0] + l[0]
        n_y = current_location[1] + l[1]

        if (map_shape[0] > n_x >= 0) and (map_shape[1] > n_y >= 0):

            n = node(parent= current_node,  #current_node,
                     location=(n_x, n_y),
                     value=obs_map[n_x][n_y])
            #print('location: ', n.location, 'value: ', n.value)

            if not obs_map[n_x, n_y] == -1:
                neighbors.append(n)

    return neighbors


def search(obs_map):

    frontier = PriorityQueue()
    already_visited = []

    #start_node = node(value=None, parent=None, location=None)
    for x, c in enumerate(obs_map):
        if -2 in c:
            start_node = node(value=-2, parent=None, location=(x, list(c).index(-2)))

    # add starting cell to open list
    frontier.add(start_node, 0)

    while not frontier.isEmpty():
        current_node = frontier.remove()
        #print('current node in search: ', current_node)
        if current_node.value == -3:
            break

        neighbours = get_neighbors(obs_map, current_node)
        #print('neighbours: ', str([x.location for x in neighbours]))
        for child_node in neighbours:
            if not child_node.location in already_visited:
                #print('child_node:', child_node)
                frontier.add(child_node, 0)
                already_visited.append(child_node.location)
            else:
                pass
                #print('already visited location: ', child_node.location)

    return current_node


if __name__ == '__main__':
    #_map_ = generateMap2d([40, 40])
    _map_, info = generateMap2d_obstacle([60, 60])

    print(_map_)
    plt.clf()
    plt.imshow(_map_)
    plt.show()

    search_result = search(_map_)
    print('Final result:', search_result)
    my_path=[]

    while not search_result.parent is None:
        my_path.append(search_result.location)
        search_result = search_result.parent
    my_path.append(search_result.location)

    my_path = np.array(my_path)
    print('camino:', my_path)

    plotMap(_map_, my_path)

