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
