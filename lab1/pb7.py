"""
Să se determine al k-lea cel mai mare element al unui șir de numere
cu n elemente (k < n). De ex. al 2-lea cel mai mare element din șirul [7,4,6,3,9,1] este 7.
"""

# Solutia mea - sortez sirul si returnez elementul de pe pozitia n - k
# COMPLEXITATE: O(n log n)
def k_element_mare(sir, k):
    return sorted(sir)[- k]

# Tests
print(k_element_mare([7,4,6,3,9,1], 2))
print(k_element_mare([7,4,6,3,9,1], 1))
print(k_element_mare([7,4,6,3,9,1], 3))

#---------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: O(n log n)
def k_element_mare_copilot(sir, k):
    sir.sort()
    return sir[- k]

# Tests
print(k_element_mare_copilot([7,4,6,3,9,1], 2))
print(k_element_mare_copilot([7,4,6,3,9,1], 1))
print(k_element_mare_copilot([7,4,6,3,9,1], 3))

# Concluzie: Ambii algoritmi au aceeasi complexitate.