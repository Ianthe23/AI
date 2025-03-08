"""
Să se determine cuvintele unui text care apar exact o singură dată în acel text. 
De ex. cuvintele care apar o singură dată în ”ana are ana are mere rosii ana" 
sunt: 'mere' și 'rosii'.
"""

# Algoritmul meu - impart sirul in cuvinte si retin intr-un dictionar
# de cate ori apare fiecare cuvant, iar apoi retin intr-o lista cuvintele
# care apar o singura data
# input - sir - string, sirul de caractere
# output - cuvinte - string[] ista cu cuvintele care apar o singura data
# COMPLEXITATE: Theta(n)
def cuvinte_unice(sir):
    words = sir.split()
    cuvinte = {}

    for word in words:
        if word in cuvinte:
            cuvinte[word] += 1
        else:
            cuvinte[word] = 1

    return [word for word in cuvinte if cuvinte[word] == 1]

# Tests
print("Teste la algoritmul meu:")
def test():
    assert cuvinte_unice("ana are ana are mere rosii ana") == ["mere", "rosii"]
    assert cuvinte_unice("ana are ana are mere rosii ana ana") == ["mere", "rosii"]

test()

#-----------------------------------------------------------------------------------------------------------------------

# Algoritmul copilotului:
# COMPLEXITATE: Theta(n)
def cuvinte_unice_copilot(sir):
    words = sir.split()
    return [word for word in words if words.count(word) == 1]

# Tests
print("Teste la algoritmul copilotului:")
def test_copilot():
    assert cuvinte_unice_copilot("ana are ana are mere rosii ana") == ["mere", "rosii"]
    assert cuvinte_unice_copilot("ana are ana are mere rosii ana ana") == ["mere", "rosii"]

test_copilot()

# Concluzie: Algoritmul copilotului este mai simplu si mai elegant, folosind
# functia count() care returneaza de cate ori apare un cuvant in lista words.
# Dar din punct de vedere al complexitatii, ambii algoritmi au aceeasi complexitate.

