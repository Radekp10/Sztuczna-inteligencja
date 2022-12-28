# Implementation of ID3 algorithm
# author: Radosław Pietkun


import numpy as np


# Nodes in tree
class Node:
    def __init__(self, value, is_leaf):
        self.value = value  # class_id if leaf or attribute_id if inner node
        self.children = {}  # children are in a dictionary: key - attribute value, value - next node (child node)
        self.is_leaf = is_leaf

    def set_child(self, attr_val, child_node):
        self.children[attr_val] = child_node

    def has_children(self):
        return not self.is_leaf


# entropy for set 'U'
def I(U, class_column_id):
    class_column = U[:, class_column_id]
    class_id, counts = np.unique(class_column, return_counts=True)
    all_counts = sum(counts)  # total number of samples in U set
    f = np.zeros(len(class_id))  # frequencies for each class
    ln_f = np.zeros(len(class_id))
    for i in range(len(class_id)):
        f[i] = counts[i]/all_counts
        ln_f[i] = np.log(f[i])
    entropy = -np.sum(np.multiply(f, ln_f))
    return entropy


# entropy for set 'U' divided by attribute 'd'
def Inf(d, U, class_column_id):
    U_subsets = split_samples(d, U)  # list of subsets (each subset is a 2D array)
    num_samples = U.shape[0]  # total number of samples in 'U'
    inf = 0  # entropy for divided set
    for i in range(len(U_subsets)):
        inf += U_subsets[i].shape[0]/num_samples * I(U_subsets[i], class_column_id)
    return inf


def find_best_attr(D, U, class_column_id):
    d = []  # list of attributes, each attribute is a dictionary with 1 key
    for key, value in D.items():
        d.append(dict.fromkeys([key], value))
    InfGain = np.zeros(len(D))
    entropy = I(U, class_column_id)
    for i in range(len(InfGain)):
        InfGain[i] = entropy - Inf(d[i], U, class_column_id)
    best_id = np.argmax(InfGain)
    return d[best_id]


# split 'U' into subsets, there will be as many subsets as many values has 'd' attribute
def split_samples(d, U):
    d_key = list(d.keys())[0]  # attribute id (number of column in 'U' set)
    d_values = list(d.values())[0]  # all possibles values for attribute 'd'
    U_subsets = []  # list of subsets of 'U'
    for val in d_values:
        chosen_rows = np.where(U[:, d_key] == val)
        U_subsets.append(U[chosen_rows])
    return U_subsets


# D - dictionary of attributes: key - attribute id (id of column in 'U'), value - list of possible values
# U - training samples (2D array)
def id3(D, U, class_column_id, max_depth):
    # no more training samples
    if U.shape[0] == 0:
        node = Node(value=None, is_leaf=True)
        return node

    class_column = U[:, class_column_id]
    class_id, counts = np.unique(class_column, return_counts=True)

    # all samples are members of one class
    if len(class_id) == 1:
        leaf = Node(value=class_id[0], is_leaf=True)
        return leaf

    # no more attributes to analyse or reached max tree depth
    if len(D) == 0 or max_depth == 0:
        best_ids = np.argwhere(counts == np.amax(counts))  # find all classes that are the most popular
        chosen_id = np.random.choice(best_ids.flatten())  # choose randomly one of them
        leaf = Node(value=class_id[chosen_id], is_leaf=True)
        return leaf

    d = find_best_attr(D, U, class_column_id)  # 'd' is a dictionary with 1 key (attr id) and 1 value (list of possible values for attribute)
    U_subsets = split_samples(d, U)  # list of 2D arrays

    # delete attribute 'd' (already used)
    d_id = list(d.keys())[0]
    D_copy = D.copy()  # copy dictionary, otherwise recursion would overwrite primary dictionary
    D_copy.pop(d_id)

    d_values = list(d.values())[0]  # all possibles values for attribute 'd'
    node = Node(value=d_id, is_leaf=False)
    for i in range(len(U_subsets)):
        child = id3(D_copy, U_subsets[i], class_column_id, max_depth-1)
        node.set_child(attr_val=d_values[i], child_node=child)
    return node

