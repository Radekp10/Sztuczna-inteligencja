# Plik z funkcjami celu oraz ich gradientami
# autor: Radosław Pietkun

import math
import numpy as np


# Pierwsza funkcja celu (f1) i jej gradient (g1)
def f1(x):
    return pow(x, 2)


def g1(x):
    return 2 * x


# Druga funkcja celu (f2) i jej gradient (g2)
def f2(x):
    return pow(x[0] + 9, 2) + pow(x[1] - 9, 2) - 5 * math.cos(10 * math.sqrt(pow(x[0] + 9, 2) + pow(x[1] - 9, 2)))


def g2(x):
    ga = 50 * math.sin(10 * math.sqrt(pow(x[0] + 9, 2) + pow(x[1] - 9, 2))) / math.sqrt(pow(x[0] + 9, 2) + pow(x[1] - 9, 2))
    g = np.array([(x[0] + 9) * (2 + ga), (x[1] - 9) * (2 + ga)])
    return g
