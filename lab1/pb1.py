"""
Să se determine ultimul (din punct de vedere alfabetic) cuvânt 
care poate apărea într-un text care conține mai multe cuvinte separate prin ” ” (spațiu). 
De ex. ultimul (dpdv alfabetic) cuvânt din ”Ana are mere rosii si galbene” este cuvântul "si".
"""

# Algoritmul meu - impart sirul in cuvinte si retin intr-o variabila 
# acel cuvant care este mai mare alfabetic decat anteriorul
# input: - sir: string, sirul de caractere
# output: - cuvant: string, ultimul cuvant alfabetic
# COMPLEXITATE: O(nlogn)
def ultim_cuvant(sir):
    words = sorted(sir.split(), reverse=True)
    return words[0]

# Tests
print("Teste la algoritmul meu:")
def test():
    assert ultim_cuvant("apple banana cherry date") == "date"
    assert ultim_cuvant("Ana are mere rosii si galbene") == "si"

test()

#-----------------------------------------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: O(nlogn)
def ultim_cuvant_copilot(sir):
    return sorted(sir.split())[-1]

# Tests
print("Teste la algoritmul copilotului:")
def test_copilot():
    assert ultim_cuvant_copilot("apple banana cherry date") == "date"
    assert ultim_cuvant_copilot("Ana are mere rosii si galbene") == "si"

test_copilot()

# Concluzie: Ambii algoritmi au aceeasi complexitate, dar algoritmul copilotului
# este mai simplu si mai elegant, folosind functia sorted() care sorteaza cuvintele
# in ordine alfabetica si apoi returneaza ultimul cuvant din lista sortata.
