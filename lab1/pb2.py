"""
Să se determine distanța Euclideană între două locații identificate 
prin perechi de numere. De ex. distanța între (1,5) și (4,1) este 5.0
"""

import math

# Solutia mea - iau din biblioteca math si prin metoda sqrt fac radical
# din (x2 - x1)^2 + (y2 - y1)^2
# COMPLEXITATE: Theta(1)
def distanta_euclidiana(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Teste
print("Teste la algoritmul meu:")
print(distanta_euclidiana(3, 4, 6, 8))
print(distanta_euclidiana(1, 5, 4, 1))
print("")

#-----------------------------------------------------------------------------------------------------------------------

# Algoritmul copilotului:
def distanta_euclidiana_copilot(x1, y1, x2, y2):
    return math.dist([x1, y1], [x2, y2])

# Teste
print("Teste la algoritmul copilotului:")
print(distanta_euclidiana_copilot(3, 4, 6, 8))
print(distanta_euclidiana_copilot(1, 5, 4, 1))

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia math.dist() care calculeaza distanta euclidiana intre doua puncte. Dar
# din punct de vedere al coimplexitatii, ambii algoritmi au aceeasi complexitate.