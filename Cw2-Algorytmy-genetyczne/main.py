# Test algorytmu genetycznego w wersji Hollanda dla wektorów binarnych
# autor: Radosław Pietkun


from holland_algorithm import *
from function import *
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':

    # Generacja populacji początkowej
    mi0 = 100  # rozmiar populacji początkowej
    n = 6  # wymiar problemu
    b = 3  # liczba bitów do kodowania 1 liczby
    P0 = np.zeros([mi0, n, b], 'i1')
    for i in range(P0.shape[0]):
        for j in range(P0.shape[1]):
            for k in range(P0.shape[2]):
                P0[i, j, k] = random.randint(0, 1)

    # Test algorytmu Hollanda
    x, g = holland_algorithm(f, P0, 100, 0.01, 0.7, 1000)
    print("Najlepszy znaleziony osobnik:")
    print(x)
    print("Max wartość funkcji celu: ", g)


    """ 
    # Wykres funkcji 2D (przy założeniu x0=x1=...=x5)
    x = np.arange(-4, 3, 0.01)
    y = -6*(x**4 -16*x**2 + 5*x)/2
    plt.plot(x, y)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    """
