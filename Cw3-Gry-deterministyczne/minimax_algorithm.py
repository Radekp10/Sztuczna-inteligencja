# Minimax algorithm implementation
# author: Radosław Pietkun

import random


def is_terminal(s):
    if not s.children:  # node has no children
        return True
    else:
        return False


def h(s):
    return s.value  # payoff or heuristic value


def find_successors(s):
    successors = []
    for i in s.children:
        successors.append(i)
    return successors


def is_max_move(s):
    return s.max_move  # check whose turn is now


def minimax(s, d):
    if is_terminal(s) or d == 0:
        return h(s), s.id
    successors = find_successors(s)
    w = []
    for u in successors:
        w_u, _ = minimax(u, d-1)
        w.append(w_u)
    if is_max_move(s):
        max_w = max(w)
        next_best_moves = []  # list of all moves that are the best (all of them are equally good)
        i = 0
        for u in successors:
            if w[i] == max_w:
                next_best_moves.append(u)
            i += 1
        chosen_move = random.randint(0, len(next_best_moves)-1)  # choose randomly 1 move from the best moves list
        return max_w, next_best_moves[chosen_move].id
    else:
        min_w = min(w)
        next_best_moves = []  # list of all moves that are the best (all of them are equally good)
        i = 0
        for u in successors:
            if w[i] == min_w:
                next_best_moves.append(u)
            i += 1
        chosen_move = random.randint(0, len(next_best_moves) - 1)
        return min_w, next_best_moves[chosen_move].id
