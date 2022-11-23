# from search_algorithm import
from path_planning import generateMap2d, generateMap2d_obstacle, plotMap
import random
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import namedtuple
import copy

import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(5000)

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


def search(obs_map, search_type):

    already_visited=[]
    frontier = PriorityQueue()

    def find_value(value):
        start_location = node(value=None, parent=None, location=None)
        for x, c in enumerate(obs_map):
            if value in c:
                start_location = node(value=value, parent=None, location=(x, list(c).index(value)))
        return start_location

    start_node = find_value(-2)
    end_node = find_value(-3)

    def get_neighbors(current_node):
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
                if not obs_map[n_x, n_y] == -1:
                    neighbors.append(n)
        return neighbors

    def euclidian_heuristic(n1, n2):
        dx = n1.location[0] - n2.location[0]
        dy = n1.location[1] - n2.location[1]
        h = np.sqrt(dx ** 2 + dy ** 2)
        return h

    def get_priority(search_type, prev_cost):
        if search_type == 'Random':
            return random.randint(-4, 4)
        if search_type == 'BSF':
            return 0
        if search_type == 'DSF':
            return prev_cost - 1
        if search_type == 'Greedy':
            p =euclidian_heuristic(child_node, end_node)
            print(p)
            return p

    def paint_map():
        # Paint the map with the searched area:
        counter = 1
        for x in already_visited[1:]:
            if obs_map[x[0]][x[1]] != -3:
                obs_map[x[0]][x[1]] = counter
            counter += 1

    def trace_back(node):
        my_path=[]
        while not node.parent is None:
            my_path.append(node.location)
            node = node.parent
        my_path.append(node.location)
        my_path = np.array(my_path)
        return my_path

    priority = 0

    # add starting cell to open list
    frontier.add(start_node, 0)
    already_visited.append(start_node.location)

    while not frontier.isEmpty():
        current_node = frontier.remove()
        if current_node.value == -3:
            break

        neighbours = get_neighbors(current_node)
        for child_node in neighbours:
            if not child_node.location in already_visited:
                priority = get_priority(search_type=search_type, prev_cost=priority)
                frontier.add(child_node, priority)
                already_visited.append(child_node.location)
            else:
                pass

    paint_map()
    my_path = trace_back(current_node)

    return my_path


if __name__ == '__main__':
    _map_, info = generateMap2d_obstacle([60, 60])
    #_map_= generateMap2d([20, 20])
    #  _map_= np.array([[-2,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,-3]])

    print(_map_)
    plt.clf()
    plt.imshow(_map_)
    plt.show()

    for search_type in ["Random", "DSF", "BFS", "Greedy"]:
        map2 = copy.copy(_map_)
        search_result = search(map2, search_type)
        plotMap(map2, search_result, search_type)

