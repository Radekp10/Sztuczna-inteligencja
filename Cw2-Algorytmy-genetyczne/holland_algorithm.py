# Implementacja algorytmu genetycznego w wersji Hollanda dla wektorów binarnych
# autor: Radosław Pietkun

import random
import numpy as np


# Funkcja do wyznaczania ocen osobników
# Parametry:
# - f - f. celu
# - Pt - populacja osobników
# Wartość zwracana:
# - G - lista ocen wszystkich osobników w populacji
def grade(f, Pt):
    G = []
    for i in range(len(Pt)):
        G.append(f(Pt[i]))
    return G



# Funkcja do znajdowania najlepszego osobnika
# Parametry:
# - Pt - populacja
# - G - wektor ocen osobników w populacji
# Wartości zwracane:
# - x - najlepszy osobnik
# - g - jego ocena
def find_best(Pt, G):
    g = max(G)  # znajdź najlepszą ocenę
    x = Pt[G.index(g)]  # znajdź pierwszego osobnika, który ma taką ocenę
    return x, g



# Funkcja do operacji reprodukcji
# Parametry:
# - Pt - populacja
# - G - wekktor ocen
# - mi - rozmiar populacji
# Wartość zwracana:
# - R - populacja po reprodukcji
def reproduction(Pt, G, mi):
    R = np.zeros([mi, Pt.shape[1], Pt.shape[2]], 'i1')
    G1 = []  # wektor przeskalowanych wartości ocen (wartości funkcji celu) do wartości <0, 1>
    ps = []  # wartości funkcji przystosowania, prawdopodobieńśtwa wyboru poszczególnych osobników
    for i in range(len(Pt)):
        if max(G) != min(G):
            G1.append((G[i]-min(G)) / (max(G) - min(G)))
        else:
            G1.append(1)
    s = sum(G1)
    for i in range(len(Pt)):
        ps.append(G1[i] / s)

    # Wybierz 'mi' osobników z populacji 'Pt' na podstawie p-ństw 'ps':
    # koło ruletki rozcięte na pasek dlugości 1, każdy osobnik ma na nim pole o długości proporcjonalnej do 'ps'
    # losujemy punkt z przedziału <0, 1> i sprawdzamy, na pole którego osobnika trafiliśmy,
    # wybrany osobnik przechodzi dalej, procedura jest powtarzana aż uzbiera się 'mi' osobników
    for i in range(mi):
        ran_num = random.uniform(0, 1)
        prob = 0  # zmienna sumująca p-nstwa, odpowiada to przesuwaniu się po kolejnych polach odcinka w prawo
        for j in range(len(Pt)):
            prob += ps[j]
            if ran_num <= prob:
                R[i] = Pt[j]
                break
    return R



# Funkcja do operacji krzyżowania i mutacji
# Parametry:
# - R - populacja po reprodukcji
# - pm - p-ństwo mutacji
# - pc - p-ństwo krzyżowania
# Wartość zwracana:
# - M - populacja po krzyżowaniu i mutacjach
def crossover_mutation(R, pm, pc):
    mi = len(R)  # liczba osobników

    C = np.zeros([mi, R.shape[1], R.shape[2]], 'i1')  # osobniki po krzyżowaniu
    M = np.zeros([mi, R.shape[1], R.shape[2]], 'i1')  # osobniki po mutacji

    # krzyżowanie
    for i in range(int(mi/2)):  # mi/2 operacji, w każdej powstaje 2 osobników w wyniku krzyżowania, zatem mi/2 * 2 = mi osobników na koniec
        parent1id = random.randint(0, len(R)-1)  # losowanie 1. rodzica (ze zwracaniem)
        parent2id = random.randint(0, len(R)-1)  # losowanie 2. rodzica (ze zwracaniem)
        if random.uniform(0, 1) < pc:
            # losowanie punktu przecięcia (locus) l
            # l wskazuje nr bitu, po którym genotyp zostanie przecięty, jest (l_wymiarów x l_bitów - 1) możliwości podziału
            l = random.randint(0, R.shape[1]*R.shape[2]-2)

            # zamiana macierzy na wektory bitów
            parent1 = R[parent1id].reshape(R.shape[1]*R.shape[2])
            parent2 = R[parent2id].reshape(R.shape[1]*R.shape[2])

            # krzyżowanie
            child1 = np.concatenate((parent1[:l+1], parent2[l+1:]))
            child2 = np.concatenate((parent2[:l+1], parent1[l+1:]))

            # zamiana wektorów z powrotem na macierze
            child1 = child1.reshape(R.shape[1], R.shape[2])
            child2 = child2.reshape(R.shape[1], R.shape[2])

            # dodanie dzieci do populacji C
            C[i] = child1
            C[int(i+mi/2)] = child2
        else:
            # dodanie rodziców do populacji C
            C[i] = R[parent1id]
            C[int(i+mi/2)] = R[parent2id]

    # mutacje
    M = C
    for i in range(C.shape[0]):  # dla każdego osobnika
        for j in range(C.shape[1]):  # dla każdego wymiaru
            for k in range(C.shape[2]):  # dla każdego bitu
                if random.uniform(0, 1) < pm:
                    if M[i, j, k]:
                        M[i, j, k] = 0
                    else:
                        M[i, j, k] = 1

    return M



# Algorytm genetyczny Hollanda
# Parametry:
# - f - f.celu
# - P0 - populacja początkowa
# - mi - rozmiar populacji
# - pm - p-ństwo mutacji
# - pc - p-ństwo krzyżowania
# - tmax - max liczba iteracji
# Wartości zwracane:
# - x - najlepszy znaleziony osobnik
# - g - maksymalna wartość f. celu
def holland_algorithm(f, P0, mi, pm=0.01, pc=0.7, tmax=1000):
    if mi % 2 == 1:
        print("Rozmiar populacji powinien być liczbą parzystą")
        return None, None
    t = 0
    Pt = P0
    G = grade(f, Pt)  # wektor ocen (ocen tyle co osobników)
    x, g = find_best(Pt, G)
    while t < tmax:
        R = reproduction(Pt, G, mi)
        M = crossover_mutation(R, pm, pc)
        G = grade(f, M)
        xt, gt = find_best(M, G)
        if gt >= g:  # sprawdzenie, czy nowy osobnik jest lepszy od poprzedniego
            g = gt
            x = xt
        Pt = M
        t += 1
    return x, g
