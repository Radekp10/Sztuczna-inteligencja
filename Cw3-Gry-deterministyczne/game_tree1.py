# Game tree class and example


class Tree:
    def __init__(self, id, value, max_move=True):
        self.id = id  # node id
        self.value = value  # node value (payoff or heuristic)
        self.children = []
        self.max_move = max_move  # whose move (MAX or MIN) is now, players play by turns

    def set_children(self, children):
        self.children.extend(children)


# Game tree example (the one from the lecture)
# Level 0
root = Tree(0, 6, True)  # if the last parameter is 'True' then MAX starts the game, otherwise MIN will start
# Level 1
node1 = Tree(1, 8, not root.max_move)
node2 = Tree(2, 5, not root.max_move)
node3 = Tree(3, 3, not root.max_move)
root.set_children([node1, node2, node3])
# Level 2
node4 = Tree(4, 4, not node1.max_move)
node5 = Tree(5, 4, not node1.max_move)
node6 = Tree(6, 2, not node2.max_move)
node7 = Tree(7, 7, not node2.max_move)
node8 = Tree(8, 5, not node3.max_move)
node9 = Tree(9, 4, not node3.max_move)
node1.set_children([node4, node5])
node2.set_children([node6, node7])
node3.set_children([node8, node9])
# Level 3
node10 = Tree(10, 7, not node4.max_move)
node11 = Tree(11, 8, not node4.max_move)
node12 = Tree(12, 5, not node5.max_move)
node13 = Tree(13, 8, not node6.max_move)
node14 = Tree(14, 4, not node6.max_move)
node15 = Tree(15, 6, not node7.max_move)
node16 = Tree(16, 7, not node8.max_move)
node17 = Tree(17, 5, not node9.max_move)
node18 = Tree(18, 7, not node9.max_move)
node4.set_children([node10, node11])
node5.set_children([node12])
node6.set_children([node13, node14])
node7.set_children([node15])
node8.set_children([node16])
node9.set_children([node17, node18])
# Level 4
node19 = Tree(19, 5, not node10.max_move)
node20 = Tree(20, 6, not node10.max_move)
node21 = Tree(21, 7, not node11.max_move)
node22 = Tree(22, 4, not node11.max_move)
node23 = Tree(23, 5, not node11.max_move)
node24 = Tree(24, 3, not node12.max_move)
node25 = Tree(25, 6, not node13.max_move)
node26 = Tree(26, 6, not node14.max_move)
node27 = Tree(27, 9, not node14.max_move)
node28 = Tree(28, 7, not node15.max_move)
node29 = Tree(29, 5, not node16.max_move)
node30 = Tree(30, 9, not node17.max_move)
node31 = Tree(31, 8, not node17.max_move)
node32 = Tree(32, 6, not node18.max_move)
node10.set_children([node19, node20])
node11.set_children([node21, node22, node23])
node12.set_children([node24])
node13.set_children([node25])
node14.set_children([node26, node27])
node15.set_children([node28])
node16.set_children([node29])
node17.set_children([node30, node31])
node18.set_children([node32])
