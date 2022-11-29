import poker_game_example as pg_
import poker_environment as pe_
import random
import numpy as np
from poker_environment import AGENT_ACTIONS, BETTING_ACTIONS
from importlib import reload


# To implement the different algorithms, change the assigned value in cost function definition.

def cost(prev_cost, last_checked_state, search_algorithm):
    # Function to calculate the cost or priority used to search the tree
    if search_algorithm == 'DFS':
        # Goes deep into the tree.
        return prev_cost - 1
    if search_algorithm == 'BFS':
        # assigns an equal priority to every node.
        return 0
    if search_algorithm == 'RANDOM':
        # assigns a random value to the priority.
        return random.randint(-4, 4)
    if search_algorithm == 'GREEDY':
        # Uses the number of times both players made a bid until now as a heuristic.
        heuristic = traceback_bidding_ammount(last_checked_state)
        #print('Total times both players have bidded:' ,traceback_bidding)
        return heuristic
    if search_algorithm == 'GREEDY_CUSTOM':
        # Uses the absolute value of the difference between the number of times both players made a bid until now
        # and the real number of biddings as a heuristic.
        # we found 33 as a somewhat close number to the actual biddings to find the result.
        traceback_bidding = traceback_bidding_ammount(last_checked_state)
        heuristic= abs(33 - traceback_bidding)

        #print('How close we are to real # of biddings:',heuristic,' ,Total times both players have bidded:', traceback_bidding)
        return heuristic

def traceback_bidding_ammount(end_state_):
    state__ = end_state_
    nn_level = 0
    number_of_biddings = []
    while state__.parent_state != None:
        nn_level += 1
        if state__.agent.action != None:
            if "BET" in state__.agent.action:
                number_of_biddings.append(['agent',state__.agent.action])
        if state__.opponent.action != None:
            if "BET" in state__.opponent.action:
                number_of_biddings.append(['opponent',state__.opponent.action])
        state__ = state__.parent_state
    return len(number_of_biddings)



def print_my_state(state):
    print('object', state,
          ',chand:', state.nn_current_hand,
          ',cbid:', state.nn_current_bidding,
          ',cphase:', state.phase,
          ',cpot:', state.pot,
          ',cactagent:', state.acting_agent,
          ',cparent:', state.parent_state,
          ',cchildren:', state.children,
          ',cagent:', state.agent,
          ',coponent:', state.opponent,
          'agentstack:', state.agent.stack,
          'opponentstack:', state.opponent.stack,
          ',cshowdowninfo:', state.showdown_info)

def game(search_algorithm):

    MAX_HANDS = 4
    INIT_AGENT_STACK = 400

    # initialize 2 agents and a game_state
    agent = pg_.PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)
    opponent = pg_.PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)

    init_state = pg_.GameState(nn_current_hand_=0,
                               nn_current_bidding_=0,
                               phase_='INIT_DEALING',
                               pot_=0,
                               acting_agent_=None,
                               agent_=agent,
                               opponent_=opponent,
                               )

    game_state_queue = []
    already_visited = []
    game_on = True
    round_init = True
    Found_solution = False
    counter = 0
    priority = 0

    while game_on:

        if len(already_visited) > 20000:
            print('Could not find a solution in 20000 steps, sorry.')
            end_state_ = current_state[0]
            game_on = False

        if round_init:
            round_init = False
            states_ = pg_.get_next_states(init_state)
            current_state = states_
            game_state_queue.append([init_state, 0])
        else:
            game_state_queue.sort(key=lambda x: x[1])
            current_state = game_state_queue.pop(0)
            if not current_state[0] in already_visited:
                next_states = pg_.get_next_states(current_state[0])
                for _state_ in next_states:
                    if _state_.nn_current_hand <= 4:
                        if _state_.agent.stack > 300:
                            _state_.parent_state = current_state[0]
                            priority = cost(priority, _state_, search_algorithm)
                            game_state_queue.append([_state_, priority])
            already_visited.append(current_state[0])
            #print_my_state(current_state[0])

            if current_state[0].phase == 'SHOWDOWN' and (current_state[0].opponent.stack <= 300):
                end_state_ = current_state[0]
                Found_solution = True
                game_on = False

    state__ = end_state_
    nn_level = 0

    print('------------ print game info ---------------')
    print('Visited nodes: ', len(already_visited))
    print('nn_states_total', len(game_state_queue))

    if Found_solution:
        while state__.parent_state != None:
            nn_level += 1
            #print(nn_level)
            #state__.print_state_info()
            state__ = state__.parent_state

        print(nn_level)
        print('Agent wins! Agent stack: ',end_state_.agent.stack ,' read the game starting from the bottom and going up.')
        print('Visited nodes: ', len(already_visited))
        print('nn_states_total', len(game_state_queue))
        print('Number of hands played:', end_state_.nn_current_hand)
        print('Number of bids in the game:', traceback_bidding_ammount(end_state_))
        nodes_expanded = len(already_visited)
        number_hands = end_state_.nn_current_hand
        number_biddings = traceback_bidding_ammount(end_state_)


    else:
        print('Could not find a solution')
        nodes_expanded = len(already_visited)
        number_hands = None
        number_biddings = None


    return nodes_expanded, number_hands, number_biddings


search_algorithms = ['DFS', 'BFS', 'RANDOM', 'GREEDY', 'GREEDY_CUSTOM']
#search_algorithms = ['DFS'] #, 'BFS', 'RANDOM', 'GREEDY', 'GREEDY_CUSTOM']

nodes = {}
hands = {}
biddings = {}

for element in search_algorithms:
    nodes[element] = []
    hands[element] = []
    biddings[element] = []

for times in range(20):
    reload(pe_)
    for element in search_algorithms:
        print(element)
        nodes_expanded, number_hands, number_biddings = game(element)

        nodes[element].append(nodes_expanded)
        if not number_hands is None:
            hands[element].append(number_hands)
        if not number_biddings is None:
            biddings[element].append(number_biddings)
        print('------------------')
        print('Number of iterations:', times)

for element in search_algorithms:
    mean_nodes = round(np.mean(nodes[element]))
    std_dev_nodes = round(np.std(nodes[element]))

    mean_hands = round(np.mean(hands[element]), 2)
    std_dev_hands = round(np.std(hands[element]), 2)


    mean_bidding = round(np.mean(biddings[element]), 2)
    std_dev_bidding = round(np.std(biddings[element]), 2)

    print()
    print('Algorithm used: ', element)
    print('Mean nodes expanded: ', mean_nodes, '±', std_dev_nodes)
    print('Mean hands played: ', mean_hands, '±', std_dev_hands)
    print('Mean # bids done in the game: ', mean_bidding, '±', std_dev_bidding)