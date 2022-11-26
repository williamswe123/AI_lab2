# HALMSTAD UNIVERSITY, AI COURSE 2022

# from search_algorithm import
from path_planning import generateMap2d, generateMap2d_obstacle, plotMap
import random
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import namedtuple
import copy
import time
import sys

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
    info = [None] * 3
    if type(obs_map) is tuple:
        info[0] = obs_map[1][0]  # y coordinate of the bottom horizontal wall/part
        info[1] = obs_map[1][1]  # y coordinate of the top horizontal wall/part
        info[2] = obs_map[1][2]  # X coordinate of the vertical wall
        obs_map = obs_map[0]

    def find_value(value):
        start_location = node(value=None, parent=None, location=None)
        for x, c in enumerate(obs_map):
            if value in c:
                start_location = node(value=value, parent=None, location=(x, list(c).index(value)))
        return start_location

    def get_neighbors(current_node):
        map_shape = obs_map.shape
        neighbors = []
        current_location = current_node.location
        # Next Nodes
        for l in [(0, 1), (0, -1), (1, 0), (-1, 0)]:

            n_x = current_location[0] + l[0]
            n_y = current_location[1] + l[1]

            if (map_shape[0] > n_x >= 0) and (map_shape[1] > n_y >= 0):

                n = node(parent=current_node,
                         location=(n_x, n_y),
                         value=obs_map[n_x][n_y])
                if not obs_map[n_x, n_y] == -1:
                    if not n.location in already_visited:
                        neighbors.append(n)
        return neighbors

    def euclidian_heuristic(n1, n2):
        dx = n1.location[0] - n2.location[0]
        dy = n1.location[1] - n2.location[1]
        h = np.sqrt(dx ** 2 + dy ** 2)
        return h

    def manhattan_heuristic(n1, n2):
        dx = n1.location[0] - n2.location[0]
        dy = n1.location[1] - n2.location[1]
        h = abs(dx) + abs(dy)
        return h

    def custom_rotated_H_heuristic():
        custom_node_up = node(value=None, parent=None, location=(info[1], start_node.location[1]))
        custom_node_down = node(value=None, parent=None, location=(info[0], start_node.location[1]))
        p = min(euclidian_heuristic(child_node, custom_node_up),
                euclidian_heuristic(child_node, custom_node_down)) + 40
        if child_node.location[1] < info[2]:
            if info[0] >= child_node.location[0] >= info[1]:
                return p
            else:
                return euclidian_heuristic(child_node, end_node)
        else:
            return euclidian_heuristic(child_node, end_node)

    def get_priority(search_type, prev_cost):
        if search_type == 'Random':
            # assigns a random value to the priority
            return random.randint(-4, 4)
        if search_type == 'BSF':
            # Breadth-first search
            # selects the shallowest frontier node from the start node for expansion
            return 0
        if search_type == 'DSF':
            # Depth-first search
            # selects the deepest frontier node from the start node for expansion
            return prev_cost - 1
        if search_type == 'UCS':
            # Uniform cost search
            # selects the lowest cost frontier node from the start node for expansion
            # the priority for a given enqueued node 'child_node' is the computed backward cost
            # or path cost from the start node to 'child_node'.
            return len(trace_back(child_node))
        if search_type == 'Greedy_euclidian':
            # Greedy search uses estimated forward cost to assign a priority
            # uses the euclidian distance as the estimated forward cost
            p = euclidian_heuristic(child_node, end_node)
            # print(p)
            return p
        if search_type == 'A*_euclidian':
            # selects the frontier node with the lowest estimated total cost for expansion
            # total cost is the sum of estimated forward cost and computed backward cost
            p = euclidian_heuristic(child_node, end_node)
            return len(trace_back(child_node)) + p

        if search_type == 'Greedy_Manhattan':
            # uses the manhattan distance as the estimated forward cost
            p = manhattan_heuristic(child_node, end_node)
            # print(p)
            return p
        if search_type == 'A*_Manhattan':
            p = manhattan_heuristic(child_node, end_node)
            return len(trace_back(child_node)) + p

        if search_type == 'Greedy_Customized_Heuristic':
            # It works as a two part heuristic function, using euclidean distances.
            # It expands towards a point further south the south wall or further north
            # the north wall (whichever is closer) and then expands towards the goal.
            if info[0] is None:
                return 0
            return custom_rotated_H_heuristic()

        if search_type == 'A*_Customized_Heuristic':
            return len(trace_back(child_node)) + custom_rotated_H_heuristic()

    def paint_map():
        # Paint the map with the searched area:
        counter = 1
        for x in already_visited[1:]:
            if obs_map[x[0]][x[1]] != -3:
                obs_map[x[0]][x[1]] = counter
            counter += 1

    def trace_back(node):
        my_path = []
        while not node.parent is None:
            my_path.append(node.location)
            node = node.parent
        my_path.append(node.location)
        my_path = np.array(my_path)
        return my_path

    start_node = find_value(-2)
    end_node = find_value(-3)
    already_visited = []
    current_node = node(value=None, parent=None, location=None)
    frontier = PriorityQueue()
    priority = 0

    # add starting cell to open list
    frontier.add(start_node, 0)

    while not frontier.isEmpty():
        current_node = frontier.remove()

        if current_node.value == -3:
            already_visited.append(current_node.location)
            break
        neighbours = get_neighbors(current_node)

        if not current_node.location in already_visited:
            for child_node in neighbours:
                priority = get_priority(search_type=search_type, prev_cost=priority)
                frontier.add(child_node, priority)

        already_visited.append(current_node.location)

    paint_map()
    my_path = trace_back(current_node)

    # print('Title:', search_type, ', Path length: ', \
    #      len(my_path), ', Expanded nodes: ', len(already_visited))

    return my_path, already_visited


def statistical_analysis(Search_Algorithms, num_iterations):
    path_length_d = {}
    time_taken_d = {}
    expanded_nodes_d = {}
    for search_type in Search_Algorithms:
        path_length_d[search_type] = []
        time_taken_d[search_type] = []
        expanded_nodes_d[search_type] = []

    for generate_map in [generateMap2d, generateMap2d_obstacle]:
        print('---------------------------------------------------')
        print('Map style:', generate_map.__name__)
        for repetition in range(num_iterations):
            _map_ = generate_map([60, 60])
            for idx, search_type in enumerate(Search_Algorithms):
                map2 = copy.deepcopy(_map_)
                if (generate_map.__name__ == 'generateMap2d') and (
                        search_type in ["Greedy_Customized_Heuristic", "A*_Customized_Heuristic"]):
                    continue
                start_time = time.process_time_ns()
                search_result, expanded_nodes = search(map2, search_type)

                total_time = time.process_time_ns() - start_time
                path_length = len(search_result)
                expanded_nodes = len(expanded_nodes)

                time_taken_d[search_type].append(total_time)
                path_length_d[search_type].append(path_length)
                expanded_nodes_d[search_type].append(expanded_nodes)

        print('Number of iterations:', num_iterations)
        for search_type in Search_Algorithms:
            if (generate_map.__name__ == 'generateMap2d') and (
                    search_type in ["Greedy_Customized_Heuristic", "A*_Customized_Heuristic"]):
                continue
            mean_time = round(np.mean(time_taken_d[search_type]) / 1000000, 2)
            std_dev_time = round(np.std(time_taken_d[search_type]) / 1000000, 2)
            mean_path = round(np.mean(path_length_d[search_type]), 2)
            std_dev_path = round(np.std(path_length_d[search_type]), 2)
            mean_exp_nodes = round(np.mean(expanded_nodes_d[search_type]), 2)
            std_dev_exp_nodes = round(np.std(expanded_nodes_d[search_type]), 2)

            print()
            print('Algorithm used: ', search_type)
            print('Mean time (ms): ', mean_time, '±', std_dev_time)
            print('Mean path length: ', mean_path, '±', std_dev_path)
            print('Mean # expanded nodes: ', mean_exp_nodes, '±', std_dev_exp_nodes)


if __name__ == '__main__':
    info = [None] * 3
    _map_ = generateMap2d_obstacle([60, 60])
    #_map_ = generateMap2d([80, 80])
    Search_Algorithms = ["Random", "DSF", "BSF", "UCS",
                         "Greedy_euclidian", "A*_euclidian",
                         "Greedy_Manhattan", "A*_Manhattan",
                         "Greedy_Customized_Heuristic",
                         "A*_Customized_Heuristic"]

    if type(_map_) is tuple:
        map_plot = _map_[0]
    else:
        map_plot = _map_

    print(map_plot)
    plt.clf()
    plt.imshow(map_plot)
    plt.show()

    number_of_maps = len(Search_Algorithms)
    if (type(_map_) is not tuple):
        number_of_maps = number_of_maps - 2

    for idx, search_type in enumerate(Search_Algorithms):
        map2 = copy.deepcopy(_map_)
        if (type(map2) is not tuple) and (search_type in ["Greedy_Customized_Heuristic", "A*_Customized_Heuristic"]):
            continue
        search_result, expanded_nodes = search(map2, search_type)
        print('Title:', search_type, ', Path length: ',
              len(search_result), ', Expanded nodes: ', len(expanded_nodes))
        if type(map2) is tuple:
            map2 = map2[0]
        plotMap(map2, search_result, search_type + '(' + str(idx + 1) + '/' + str(number_of_maps) + ')')

    statistical_analysis(Search_Algorithms,100)
