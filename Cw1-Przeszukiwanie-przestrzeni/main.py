# Implementacja algorytmu gradientu prostego
# autor: Radosław Pietkun


from functions import *


""" 
Funkcja implementująca algorytm gradientu prostego.
Parametry funkcji:
f - funkcja celu
g - gradient funkcji celu
x0 - punkt startowy
bt - długość kroku
a - współczynnik redukcji , wartość domyślna 0.5
k - limit liczby redukcji kroku (podczas jednej iteraci algorytmu), wartość domyślna 6
eps - dokładność wyznaczenia minimum, wartość domyślna 0.01 
Wartości zwracane:
f(x) - wyznaczona wartość minimum
counter - liczba wykonanych iteracji algorytmu
"""
def gradient_descent(f, g, x0, bt, eps=0.01, a=0.5, k=6):
    x = np.array(x0)
    step_reduction_counter = 0  # licznik redukcji kroku
    counter = 0  # licznik iteracji algorytmu
    print("t=%d, x=%s, f(x)=%.2f, |g(x)|=%.2f" % (counter, x, f(x), np.linalg.norm(g(x))))
    while np.linalg.norm(g(x)) >= eps:
        d = -g(x)
        x1 = x + d * bt  # wyznaczenie nastepnego punktu
        if f(x1) >= f(x):
            if step_reduction_counter < k:
                bt *= a  # zmniejszenie dlugości kroku
                step_reduction_counter += 1
                print("Zmniejszono krok")
                continue
            else:
                print("Zatrzymanie algorytmu z powodu przekroczenia limitu liczby redukcji kroku!")
                return None, None

        step_reduction_counter = 0  # wyzerowanie licznika redukcji kroku
        counter += 1
        x = x1
        print("t=%d, x=%s, f(x)=%.2f, |g(x)|=%.2f" % (counter, x, f(x), np.linalg.norm(g(x))))
    return f(x), counter




if __name__ == '__main__':

    print("Funkcja nr 1:")
    f1min, count1 = gradient_descent(f1, g1, 100, 0.7)
    print("Liczba kroków algorytmu: %d" % count1)
    print("Wyznaczona wartość minimum funkcji: %.2f" % f1min)
    print()

    print("Funkcja nr 2:")
    f2min, count2 = gradient_descent(f2, g2, np.array([0, 0]), 0.1)
    if f2min is not None and count2 is not None:
        print("Liczba kroków algorytmu: %d" % count2)
        print("Wyznaczona wartość minimum funkcji: %.2f" % f2min)
