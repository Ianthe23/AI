"""
Pentru un șir cu n numere întregi care conține și duplicate, 
să se determine elementul majoritar (care apare de mai mult de n / 2 ori). 
De ex. 2 este elementul majoritar în șirul [2,8,7,2,2,5,2,3,1,2,2].
"""

# Algoritmul meu - parcurg sirul cu elemente unice si verific daca numarul de aparitii
# ale fiecarui element este mai mare decat n / 2
# input - sir: lista de numere
# output - numar: elementul majoritar
# COMPLEXITATE: Theta(n)
def element_majoritar(sir):
    sir_unic = list(set(sir))
    lungime = len(sir)

    for numar in sir_unic:
        if sir.count(numar) > lungime // 2:
            return numar
        
    return -1

def main(sir):
    sol = element_majoritar(sir)

    if sol != -1:
        print(sol)
    else:
        print("Nu exista un astfel de numar!")

# Tests
print("Pentru algoritmul meu:")
main([2,8,7,2,2,5,2,3,1,2,2])
print("")

#---------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: Theta(n)
def element_majoritar_copilot(sir):
    lungime = len(sir)

    for numar in sir:
        if sir.count(numar) > lungime // 2:
            return numar
        
    return -1

def main_copilot(sir):
    sol = element_majoritar_copilot(sir)

    if sol != -1:
        print(sol)
    else:
        print("Nu exista un astfel de numar!")

# Tests
print("Pentru algoritmul copilotului:")
main_copilot([2,8,7,2,2,5,2,3,1,2,2])

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia count() care returneaza de cate ori apare un element in lista sir.
# Dar din punct de vedere al complexitatii, ambii algoritmi au aceeasi complexitate.
        