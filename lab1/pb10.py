"""
Considerându-se o matrice cu n x m elemente binare (0 sau 1) sortate crescător 
pe linii, să se identifice indexul liniei care conține cele mai multe elemente de 1.
"""

# Algoritmul meu - parcurg matricea si retin indexul liniei care contine cele mai multe elemente de 1
# input - matrice: matricea de numere
# output - numar: indexul liniei care contine cele mai multe elemente de 1
# COMPLEXITATE: Theta(n * m)
def linie_maxim(matrice):
    nr_linii = len(matrice)
    maxim_suma_pe_linie = 0
    linie_maxima = -1

    for i in range (nr_linii):
        if sum(matrice[i]) > maxim_suma_pe_linie:
            maxim_suma_pe_linie = sum(matrice[i])
            linie_maxima = i

    return linie_maxima

# Tests
print("Pentru algoritmul meu:")
def test():
    assert linie_maxim([[0, 0, 1], [0, 1, 1], [1, 1, 1]]) == 2
    assert linie_maxim([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == -1

test()


#---------------------------------------------------------------------------------------

# Algoritmul copilotului:

# COMPLEXITATE: Theta(n * m)

def linie_maxim_copilot(matrice):
    return max(range(len(matrice)), key=lambda i: sum(matrice[i]))

# Tests
print("Pentru algoritmul copilotului:")
def test_copilot():
    assert linie_maxim_copilot([[0, 0, 1], [0, 1, 1], [1, 1, 1]]) == 2
    assert linie_maxim_copilot([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 0

test_copilot()


# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia sum() care calculeaza suma elementelor de pe fiecare linie a matricei.
# Dar din punct de vedere al complexitatii, ambii algoritmi au aceeasi complexitate.
        