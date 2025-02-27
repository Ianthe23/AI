"""
Să se genereze toate numerele (în reprezentare binară) cuprinse între 1 și n. 
De ex. dacă n = 4, numerele sunt: 1, 10, 11, 100.
"""

# Solutia mea - generez numerele de la 1 la n si le retin in lista numbers_bin
# COMPLEXITATE: Theta(n)
def reprez_binom(n):
    numbers_bin = []
    number = 1

    while number <= n:
        numbers_bin.append(bin(number))
        number += 1

    return numbers_bin

# 
def main(n):
    numbers = reprez_binom(n)
    for number in numbers:
        print(number)

# Tests

print("Teste la algoritmul meu:")
main(4)
print("")
main(10)
print("")

#---------------------------------------------------------------------------------------

# Algoritmul copilotului
def reprez_binom_copilot(n):
    return [bin(i) for i in range(1, n + 1)]

def main_copilot(n):
    numbers = reprez_binom_copilot(n)
    for number in numbers:
        print(number)

# Tests
print("Teste la algoritmul copilotului:")
main_copilot(4)
print("")
main_copilot(10)
print("")

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia bin() care returneaza reprezentarea binara a unui numar. Dar din punct
# de vedere al complexitatii, ambii algoritmi au aceeasi