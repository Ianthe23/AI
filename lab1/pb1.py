"""
Să se determine ultimul (din punct de vedere alfabetic) cuvânt 
care poate apărea într-un text care conține mai multe cuvinte separate prin ” ” (spațiu). 
De ex. ultimul (dpdv alfabetic) cuvânt din ”Ana are mere rosii si galbene” este cuvântul "si".
"""

# Algoritmul meu - impart sirul in cuvinte si retin intr-o variabila 
# acel cuvant care este mai mare alfabetic decat anteriorul
# COMPLEXITATE: Theta(n)
def ultim_cuvant(sir):
    last_word = ""
    words = sir.split()

    for word in words:
        if word > last_word:
            last_word = word

    return last_word

# Tests
print("Teste la algoritmul meu:")
print(ultim_cuvant("apple banana cherry date"))
print(ultim_cuvant("Ana are mere rosii si galbene"))
print("")

#-----------------------------------------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: Theta(n)
def ultim_cuvant_copilot(sir):
    return sorted(sir.split())[-1]

# Tests
print("Teste la algoritmul copilotului:")
print(ultim_cuvant_copilot("apple banana cherry date"))
print(ultim_cuvant_copilot("Ana are mere rosii si galbene"))

# Concluzie: Ambii algoritmi au aceeasi complexitate, dar algoritmul copilotului
# este mai simplu si mai elegant, folosind functia sorted() care sorteaza cuvintele
# in ordine alfabetica si apoi returneaza ultimul cuvant din lista sortata.
