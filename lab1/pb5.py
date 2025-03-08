"""
Pentru un șir cu n elemente care conține valori din mulțimea {1, 2, ..., n - 1} 
astfel încât o singură valoare se repetă de două ori, să se identifice acea valoare 
care se repetă. De ex. în șirul [1,2,3,4,2] valoarea 2 apare de două ori.
"""

# Algoritmul meu - parcurg sirul si retin intr-un dictionar de cate ori apare
# fiecare element, iar apoi returnez elementul care apare de doua ori
# input - sir: lista de numere
# output - numar: numarul care se repeta
# COMPLEXITATE: Theta(n)
def valoare_repetata(sir):
    dict = {}

    for numar in sir:
        if not numar in dict:
            dict[numar] = 1
        else:
            return numar
        
    return -1

def main(sir):
    sol = valoare_repetata(sir)

    if sol != -1:
        print(sol)
    else:
        print("Nu exista astfel de valoare!")


# Tests
print("Pentru algoritmul meu:")
def test():
    assert valoare_repetata([1, 2, 3, 4, 2]) == 2
    assert valoare_repetata([1, 2, 3, 4, 5]) == -1

test()

#---------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: Theta(n)
def valoare_repetata_copilot(sir):
    for numar in sir:
        if sir.count(numar) == 2:
            return numar
        
    return -1

def main_copilot(sir):
    sol = valoare_repetata_copilot(sir)

    if sol != -1:
        print(sol)
    else:
        print("Nu exista astfel de valoare!")

# Tests
print("Pentru algoritmul copilotului:")
def test_copilot():
    assert valoare_repetata_copilot([1, 2, 3, 4, 2]) == 2
    assert valoare_repetata_copilot([1, 2, 3, 4, 5]) == -1

test_copilot()

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia count() care returneaza de cate ori apare un element in lista sir.
# Dar din punct de vedere al complexitatii, ambii algoritmi au aceeasi complexitate.