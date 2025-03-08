"""
Să se determine produsul scalar a doi vectori rari care conțin numere reale. 
Un vector este rar atunci când conține multe elemente nule. 
Vectorii pot avea oricâte dimensiuni. 
De ex. produsul scalar a 2 vectori unisimensionali [1,0,2,0,3] și 
[1,2,0,3,1] este 4.
"""

# Solutia mea - parcurg vectorii si inmultesc elementele de pe aceleasi pozitii
# daca ambele elemente sunt nenule
# input - v1, v2: int, int, vectorii rari
# output - produs: int, produsul scalar al vectorilor
# COMPLEXITATE: Theta(n)
def produs_scalar_sparse(v1, v2):
    produs = 0
    i = 0

    length = min(len(v1), len(v2))
    while i < length:
        if v1[i] and v2[i]:
            produs += v1[i] * v2[i]
        i += 1

    return produs

# Tests
print("Teste la algoritmul meu:")
def test():
    assert produs_scalar_sparse([1,0,2,0,3], [1,2,0,3,1]) == 4
    assert produs_scalar_sparse([1,2,3], [4,5,6]) == 32

test()

#-----------------------------------------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: Theta(n)
def produs_scalar_sparse_copilot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2) if x and y)

# Tests
print("Teste la algoritmul copilotului:")
def test_copilot():
    assert produs_scalar_sparse_copilot([1,0,2,0,3], [1,2,0,3,1]) == 4
    assert produs_scalar_sparse_copilot([1,2,3], [4,5,6]) == 32

test_copilot()

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia zip() care returneaza o lista de tupluri, unde primul element din fiecare tuplu
# contine elementele de pe aceeasi pozitie din vectorii v1 si v2, iar al doilea element
# contine produsul elementelor respective. Dar din punct de vedere al complexitatii,
# ambii algoritmi au aceeasi complexitate.