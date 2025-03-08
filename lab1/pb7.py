"""
Să se determine al k-lea cel mai mare element al unui șir de numere
cu n elemente (k < n). De ex. al 2-lea cel mai mare element din șirul [7,4,6,3,9,1] este 7.
"""

# Solutia mea - sortez sirul si returnez elementul de pe pozitia n - k
# input - sir: lista de numere
#        - k: pozitia elementului
# output - numar: al k-lea cel mai mare element
# COMPLEXITATE: O(n log n)
def k_element_mare(sir, k):
    return sorted(sir)[- k]

# Tests
print("Pentru algoritmul meu:")
def test():
    assert k_element_mare([7,4,6,3,9,1], 2) == 7
    assert k_element_mare([7,4,6,3,9,1], 1) == 9
    assert k_element_mare([7,4,6,3,9,1], 3) == 6

test()

#---------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: O(n log n)
def k_element_mare_copilot(sir, k):
    sir.sort()
    return sir[- k]

# Tests
print("Pentru algoritmul copilotului:")
def test_copilot():
    assert k_element_mare_copilot([7,4,6,3,9,1], 2) == 7
    assert k_element_mare_copilot([7,4,6,3,9,1], 1) == 9
    assert k_element_mare_copilot([7,4,6,3,9,1], 3) == 6

test_copilot()

# Concluzie: Ambii algoritmi au aceeasi complexitate.