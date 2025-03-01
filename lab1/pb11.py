"""
Considerându-se o matrice cu n x m elemente binare (0 sau 1), să se înlocuiască cu 1 
toate aparițiile elementelor egale cu 0 care sunt complet înconjurate de 1.
"""

# Algoritmul meu - parcurg matricea si verific daca elementul curent este 0 si 
# daca este complet inconjurat de 1
# COMPLEXITATE: Theta(n * m)

def inlocuire(matrice):
    nr_linii = len(matrice)
    nr_coloane = len(matrice[0])
    
    for i in range(nr_linii):
        for j in range(nr_coloane):
            if matrice[i][j] == 0 and i > 0 and i < nr_linii - 1 and j > 0 and j < nr_coloane - 1:
                if matrice[i - 1][j] == 1 and matrice[i + 1][j] == 1 and matrice[i][j - 1] == 1 and matrice[i][j + 1] == 1:
                    matrice[i][j] = 1
    
    return matrice

# Tests
print("Pentru algoritmul meu:")
print("Matricea initiala: ")
print([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
print("Matricea finala: ")
print(inlocuire([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
print("")

print("Matricea initiala: ")
print([[1, 1, 1], [1, 0, 1], [1, 0, 1]])
print("Matricea finala: ")
print(inlocuire([[1, 1, 1], [1, 0, 1], [1, 0, 1]]))
print("")

print("Matricea initiala: ")
print([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
print("Matricea finala: ")
print(inlocuire([[1, 1, 1], [0, 0, 0], [1, 1, 1]]))
print("")

#---------------------------------------------------------------------------------------

# Algoritmul copilotului:

# COMPLEXITATE: Theta(n * m)

def inlocuire_copilot(matrice):
    nr_linii = len(matrice)
    nr_coloane = len(matrice[0])
    
    for i in range(1, nr_linii - 1):
        for j in range(1, nr_coloane - 1):
            if matrice[i][j] == 0 and all(matrice[i + x][j + y] == 1 for x, y in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                matrice[i][j] = 1

    return matrice

# Tests
print("Pentru algoritmul copilotului:")
print("Matricea initiala: ")
print([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
print("Matricea finala: ")
print(inlocuire_copilot([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
print("")

print("Matricea initiala: ")
print([[1, 1, 1], [1, 0, 1], [1, 0, 1]])
print("Matricea finala: ")
print(inlocuire_copilot([[1, 1, 1], [1, 0, 1], [1, 0, 1]]))
print("")

print("Matricea initiala: ")
print([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
print("Matricea finala: ")
print(inlocuire_copilot([[1, 1, 1], [0, 0, 0], [1, 1, 1]]))

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia all() care verifica daca toate elementele dintr-un iterator sunt adevarate.
# Dar din punct de vedere al complexitatii, ambii algoritmi au aceeasi complexitate.