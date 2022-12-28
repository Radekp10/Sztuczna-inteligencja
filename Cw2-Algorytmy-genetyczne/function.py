# Funkcja celu i kodowanie liczb
# autor: Radosław Pietkun

import numpy as np


# Kodowanie U2 dla liczb <-4,3>
coding = {
    str(np.array([0, 0, 0])): 0,
    str(np.array([0, 0, 1])): 1,
    str(np.array([0, 1, 0])): 2,
    str(np.array([0, 1, 1])): 3,
    str(np.array([1, 0, 0])): -4,
    str(np.array([1, 0, 1])): -3,
    str(np.array([1, 1, 0])): -2,
    str(np.array([1, 1, 1])): -1
}


# Funkcja celu f(x), działa zarówno dla liczb binarnych jak i dla liczb dziesiętnych:
# x - tablica binarna o wymiarach: (wymiar zadania) x (liczba bitów na liczbę)
# lub wektor liczb dziesiętnych o długości: (wymiar zadania)
def f(x):
    if len(x) != 6:
        print("Błąd! Funkcja musi przyjąć wektor 6 wartości")
        return None

    # Jeśli drugi wymiar jest niepusty, to znaczy, że wektor jest w postaci binarnej
    # i konieczna jest konwersja na liczby dziesiętne;
    # jeśli wektor jest już w postaci dziesiętnej, to konwersja nie jest konieczna
    if x[0].shape:
        x_dec = []  # lista na argumenty funkcji w postaci dziesiętnej: x0, x1, ..., x5
        for i in range(len(x)):
            x_dec.append(coding[str(x[i])])  # konwersja wektora binarnego na liczbę dziesiętną
    else:
        x_dec = x

    # Wyznacznie wartości funkcji celu
    sum = 0
    for i in range(6):
        sum += x_dec[i] ** 4 - 16 * x_dec[i] ** 2 + 5 * x_dec[i]
    return -sum/2
