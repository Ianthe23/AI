"""
Să se genereze toate numerele (în reprezentare binară) cuprinse între 1 și n. 
De ex. dacă n = 4, numerele sunt: 1, 10, 11, 100.
"""

# Solutia mea - generez numerele de la 1 la n si le retin in lista numbers_bin
# input - n: numarul pana la care se genereaza numerele
# output - numbers_bin: lista cu numerele generate
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
def test():
    assert reprez_binom(4) == ['0b1', '0b10', '0b11', '0b100']
    assert reprez_binom(10) == ['0b1', '0b10', '0b11', '0b100', '0b101', '0b110', '0b111', '0b1000', '0b1001', '0b1010']

test()

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
def test_copilot():
    assert reprez_binom_copilot(4) == ['0b1', '0b10', '0b11', '0b100']
    assert reprez_binom_copilot(10) == ['0b1', '0b10', '0b11', '0b100', '0b101', '0b110', '0b111', '0b1000', '0b1001', '0b1010']

test_copilot()

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia bin() care returneaza reprezentarea binara a unui numar. Dar din punct
# de vedere al complexitatii, ambii algoritmi au aceeasi