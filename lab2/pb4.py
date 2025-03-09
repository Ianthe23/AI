import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

def load_data(filename):
    if not os.path.isfile(filename):
        print(f"File {filename} not found.")
        return None
    return pd.read_csv(filename)

# Load data
filename = 'surveyDataSience.csv'
df = load_data(filename)

# Transformarea vechimii în programare în ani (media intervalului)
df['years_in_programming'] = None
for i in range(len(df)):
    if pd.isna(df['Q6'][i]):
        continue
    if 'years' in df['Q6'][i]:
        if '-' in df['Q6'][i]:
            start, end = df['Q6'][i].replace(' years', '').split('-')
            middle = (int(start) + int(end)) / 2
            df['years_in_programming'][i] = middle
        elif '+' in df['Q6'][i]:
            start = df['Q6'][i].replace(' years', '').replace('+', '')
            df['years_in_programming'][i] = int(start) + 5  # Estimare
        elif '<' in df['Q6'][i]:
            df['years_in_programming'][i] = 0.5  # Pentru <1 an

# 1. Min-Max Scaling pentru durata studiilor și vechimea în programare
scaler = MinMaxScaler()

# Normalizarea anilor de studii
df['normalized_years_of_studies'] = scaler.fit_transform(df[['years_in_programming']])

# 2. Z-score standardization pentru durata studiilor și vechimea în programare
scaler = StandardScaler()

# Normalizarea anilor de studii
df['standardized_years_of_studies'] = scaler.fit_transform(df[['years_in_programming']])

# 3. Aplicarea pe durata studiilor universitare
# Inlocuirea valorilor de studii în ani
df['normalized_years_of_studies'] = df['years_in_programming']

# Print pentru a verifica rezultatele
print(df[['years_in_programming', 'normalized_years_of_studies', 'standardized_years_of_studies']].head())

#----------------------------------------------

# 2. Normalize pixel values in images
def normalize_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_folder, image_file), cv2.IMREAD_GRAYSCALE)
        img_normalized = img / 255.0  # Normalize to [0,1]
        
        plt.figure(figsize=(6,3))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original")
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_normalized, cmap='gray')
        plt.title("Normalized")
        plt.show()

#----------------------------------------------

# 3. Normalize word frequencies in text
def normalize_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sentences = sent_tokenize(text)
    word_frequencies = [Counter(word_tokenize(sent)) for sent in sentences]
    
    max_count = max(max(freq.values()) for freq in word_frequencies if freq)
    
    normalized_word_frequencies = [
        {word: count / max_count for word, count in freq.items()}
        for freq in word_frequencies
    ]
    return normalized_word_frequencies


normalize_images('data/images')
word_frequencies = normalize_text('data/texts.txt')
print(word_frequencies)
