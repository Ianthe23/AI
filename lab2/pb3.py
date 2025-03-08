import nltk 
import re
from unidecode import unidecode
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

file_path = './data/texts.txt'
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# 1. Count the number of sentences
sentences = sent_tokenize(text)
num_sentences = len(sentences)

# 2. Count the number of words
words = word_tokenize(text)
words = [word for word in words if word.isalnum()]  # Remove punctuation
num_words = len(words)

# 3. Count the number of unique words
unique_words = set(words)
num_unique_words = len(unique_words)

# 4. Find the shortest and longest words
shortest_word = min(unique_words, key=len)
longest_word = max(unique_words, key=len)

# 5. Remove diacritics from text
text_no_diacritics = unidecode(text)

# 6. Find synonyms for the longest word (in English, as WordNet doesn't support Romanian)
synonyms = set()
for syn in wordnet.synsets(longest_word, lang='eng'):
    for lemma in syn.lemmas():
        synonyms.add(lemma.name())

# Display results
print(f"Number of sentences: {num_sentences}")
print(f"Number of words: {num_words}")
print(f"Number of unique words: {num_unique_words}")
print(f"Shortest word: {shortest_word}")
print(f"Longest word: {longest_word}")
print(f"Longest word synonyms: {', '.join(synonyms)}")
print(f"Text without diacritics:\n{text_no_diacritics}")

