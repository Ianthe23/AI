"""
Considerându-se o matrice cu n x m elemente întregi și o listă cu perechi formate 
din coordonatele a 2 căsuțe din matrice ((p,q) și (r,s)), 
să se calculeze suma elementelor din sub-matricile identificate de fieare pereche.
"""

# Algoritmul meu - parcurg lista de perechi si pentru fiecare pereche
# calculez suma elementelor din submatricea respectiva
# COMPLEXITATE: O(n * m * p)
def suma_submatrici(matrice, perechi):
    rezultat = []
    
    for pereche in perechi:
        p, q, r, s = pereche
        suma = 0
        
        for i in range(p, r + 1):
            for j in range(q, s + 1):
                suma += matrice[i][j]
                
        rezultat.append(suma)
        
    return rezultat

# Tests
print("Pentru algoritmul meu:")
print(suma_submatrici([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [(0, 0, 1, 1), (1, 1, 2, 2)]))
print("")

#---------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: O(n * m * p)
def suma_submatrici_copilot(matrice, perechi):
    return [sum(matrice[i][j] for i in range(p, r + 1) for j in range(q, s + 1)) for p, q, r, s in perechi]

# Tests
print("Pentru algoritmul copilotului:")
print(suma_submatrici_copilot([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [(0, 0, 1, 1), (1, 1, 2, 2)]))

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# o lista de comprensie care calculeaza suma elementelor din submatricea
# identificata de fiecare pereche. Dar din punct de vedere al complexitatii,
# ambii algoritmi au aceeasi complexitate.