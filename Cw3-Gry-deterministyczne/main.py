# Interactive game simulation, minimax algorithm used
# author: Radosław Pietkun

from minimax_algorithm import *
from game_tree1 import *


# Game simulation
current_node = root  # the node where the game starts (imported from game_tree1.py)
player_mode = 1  # [1] - player is MAX, [0] - player is MIN
d = 4  # searching depth (how many future moves to analyse)

while not is_terminal(current_node):

    if player_mode == 1:  # max player mode
        if current_node.max_move:  # max player's move
            print("Player's (MAX) move...")
            possible_moves = []
            for i in current_node.children:
                possible_moves.append(i.id)
            print("Possible moves: ", possible_moves)
            value, best_move = minimax(current_node, d)
            print("Value to reach: {}, Recommended move: {}".format(value, best_move))
            try:
                move = int(input("Make a move (choose next node): "))
            except:
                print("INVALID INPUT DATA !!! Please try again\n")
                continue
            j = 0
            move_successful = False
            for i in current_node.children:
                if i.id == move:
                    current_node = current_node.children[j]  # move to next node
                    move_successful = True
                    break
                j += 1
            if not move_successful:
                print("INVALID MOVE !!! Please try again\n")
            else:
                print("You moved to node: %d\n" % current_node.id)
            continue
        else:  # min player's move
            print("Opponent's (MIN) move...")  # opponent plays optimally
            value, best_move = minimax(current_node, 1)  # d=1, opponent analyses only the next move and (doesn't look in the future)
            j = 0
            for i in current_node.children:
                if i.id == best_move:
                    current_node = current_node.children[j]  # move to next node
                    break
                j += 1
            print("Opponent moved to node: %d\n" % best_move)


    elif player_mode == 0:  # min player mode
        if current_node.max_move:  # max player's move
            print("Opponent's (MAX) move...")  # opponent plays optimally
            value, best_move = minimax(current_node, 1)  # d=1, opponent analyses only the next move (doesn't look in the future)
            j = 0
            for i in current_node.children:
                if i.id == best_move:
                    current_node = current_node.children[j]  # move to next node
                    break
                j += 1
            print("Opponent moved to node: %d\n" % best_move)
            continue
        else:  # min player's move
            print("Player's (MIN) move...")
            possible_moves = []
            for i in current_node.children:
                possible_moves.append(i.id)
            print("Possible moves: ", possible_moves)
            value, best_move = minimax(current_node, d)
            print("Value to reach: {}, Recommended move: {}".format(value, best_move))
            try:
                move = int(input("Make a move (choose next node): "))
            except:
                print("INVALID INPUT DATA !!! Please try again\n")
                continue
            j = 0
            move_successful = False
            for i in current_node.children:
                if i.id == move:
                    current_node = current_node.children[j]  # move to next node
                    move_successful = True
                    break
                j += 1
            if not move_successful:
                print("INVALID MOVE !!! Please try again\n")
            else:
                print("You moved to node: %d\n" % current_node.id)
            continue

# End of game
score = h(current_node)
print("Game ended. Node {} is terminal. Final score: {}".format(current_node.id, score))
