import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import gensim.downloader as gensim_api

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# The sentiment message we need to classify
MESSAGE_TO_CLASSIFY = "By choosing a bike over a car, I'm reducing my environmental footprint. Cycling promotes eco-friendly transportation, and I'm proud to be part of that movement."

# Azure Text Analytics setup
def analyze_with_azure():
    print("\n=== Azure Text Analytics Sentiment Analysis ===")
    
    # Replace with your Azure Text Analytics key and endpoint
    key = input("Enter your Azure Text Analytics key: ")
    endpoint = input("Enter your Azure Text Analytics endpoint: ")
    
    try:
        # Create a client for the Text Analytics service
        credential = AzureKeyCredential(key)
        client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
        
        # Analyze sentiment
        documents = [MESSAGE_TO_CLASSIFY]
        response = client.analyze_sentiment(documents=documents)[0]
        
        print(f"\nDocument Sentiment: {response.sentiment}")
        print(f"Overall confidence scores: Positive={response.confidence_scores.positive:.2f}, "
              f"Neutral={response.confidence_scores.neutral:.2f}, "
              f"Negative={response.confidence_scores.negative:.2f}")
        
        return response.sentiment, response.confidence_scores
    
    except Exception as e:
        print(f"Error with Azure Text Analytics: {e}")
        return None, None


# Text preprocessing functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into text
    return ' '.join(filtered_tokens)


# Feature extraction methods
def extract_bow_features(train_texts, test_texts):
    print("\nExtracting Bag of Words features...")
    vectorizer = CountVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    return X_train, X_test, vectorizer


def extract_tfidf_features(train_texts, test_texts):
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    return X_train, X_test, vectorizer


def extract_word2vec_features(train_texts, test_texts):
    print("\nExtracting Word2Vec features...")
    # Load pre-trained Word2Vec model
    try:
        w2v_model = gensim_api.load("word2vec-google-news-300")
        
        def text_to_vec(text):
            words = text.split()
            word_vecs = [w2v_model[word] for word in words if word in w2v_model]
            if len(word_vecs) == 0:
                return np.zeros(300)
            return np.mean(word_vecs, axis=0)
        
        X_train = np.array([text_to_vec(text) for text in train_texts])
        X_test = np.array([text_to_vec(text) for text in test_texts])
        
        return X_train, X_test, w2v_model
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return None, None, None


# Neural Network model
def build_neural_network(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_and_evaluate_model(X_train, y_train, X_test, y_test, feature_type):
    print(f"\n=== Training Neural Network with {feature_type} features ===")
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    model = build_neural_network(input_dim, num_classes)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Predict on test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy ({feature_type})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss ({feature_type})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'training_history_{feature_type}.png')
    
    return model


def analyze_message_with_model(model, vectorizer, message, feature_type):
    print(f"\n=== Analyzing message with Neural Network ({feature_type}) ===")
    
    # Preprocess the message
    processed_message = preprocess_text(message)
    
    # Transform message based on feature type
    if feature_type == "Word2Vec":
        # For Word2Vec we use the defined function from the model
        words = processed_message.split()
        word_vecs = [vectorizer[word] for word in words if word in vectorizer]
        if len(word_vecs) == 0:
            X_message = np.zeros((1, 300))
        else:
            X_message = np.mean(word_vecs, axis=0).reshape(1, -1)
    else:
        # For BoW and TF-IDF we use the vectorizer
        X_message = vectorizer.transform([processed_message]).toarray()
    
    # Predict sentiment
    prediction = model.predict(X_message)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map prediction back to sentiment label
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_sentiment = sentiment_mapping.get(predicted_class, f"unknown ({predicted_class})")
    
    confidence = prediction[0][predicted_class]
    
    print(f"Predicted sentiment: {predicted_sentiment}")
    print(f"Confidence: {confidence:.4f}")
    
    return predicted_sentiment, confidence


def main():
    print("Facebook Sentiment Analysis System")
    print("==================================")
    
    # Try to load a CSV file with emotions/sentiment data
    file_path = input("Enter the path to your sentiment dataset CSV file (default: data/review_mixed.csv): ")
    if not file_path:
        file_path = "data/review_mixed.csv"
    
    try:
        print(f"\nLoading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Display dataset info
        print(f"Dataset shape: {df.shape}")
        print("\nSample data:")
        print(df.head())
        
        # Check columns in the dataset
        text_col = input("\nEnter the name of the column containing text (default: text): ")
        if not text_col or text_col not in df.columns:
            text_col = "text" if "text" in df.columns else df.columns[0]
            
        label_col = input(f"Enter the name of the column containing sentiment labels (default: sentiment): ")
        if not label_col or label_col not in df.columns:
            label_col = "sentiment" if "sentiment" in df.columns else df.columns[-1]
        
        print(f"\nUsing '{text_col}' as text and '{label_col}' as sentiment labels")
        
        # Preprocess the text data
        print("\nPreprocessing text data...")
        df['processed_text'] = df[text_col].astype(str).apply(preprocess_text)
        
        # Encode the sentiment labels
        label_encoder = LabelEncoder()
        df['encoded_sentiment'] = label_encoder.fit_transform(df[label_col])
        
        # Display the label mapping
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        print("\nSentiment label mapping:")
        for idx, label in label_mapping.items():
            print(f"  {idx}: {label}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'].values, 
            df['encoded_sentiment'].values,
            test_size=0.2,
            random_state=42
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Feature extraction
        feature_type = input("\nChoose feature extraction method (1: Bag of Words, 2: TF-IDF, 3: Word2Vec) [default: 2]: ")
        
        if feature_type == "1":
            X_train_features, X_test_features, vectorizer = extract_bow_features(X_train, X_test)
            feature_name = "Bag of Words"
        elif feature_type == "3":
            X_train_features, X_test_features, vectorizer = extract_word2vec_features(X_train, X_test)
            feature_name = "Word2Vec"
        else:
            X_train_features, X_test_features, vectorizer = extract_tfidf_features(X_train, X_test)
            feature_name = "TF-IDF"
        
        # Train and evaluate neural network
        model = train_and_evaluate_model(X_train_features, y_train, X_test_features, y_test, feature_name)
        
        # Analyze the given message using our model
        neural_network_sentiment, nn_confidence = analyze_message_with_model(
            model, 
            vectorizer, 
            MESSAGE_TO_CLASSIFY, 
            feature_name
        )
        
        # Analyze the message using Azure
        use_azure = input("\nDo you want to analyze the message using Azure Text Analytics? (y/n): ")
        if use_azure.lower() == 'y':
            azure_sentiment, azure_confidence = analyze_with_azure()
            
            print("\n=== Comparison of Results ===")
            print(f"Message: \"{MESSAGE_TO_CLASSIFY}\"")
            print(f"Neural Network ({feature_name}): {neural_network_sentiment} (confidence: {nn_confidence:.4f})")
            if azure_sentiment:
                print(f"Azure Text Analytics: {azure_sentiment}")
                
                # Determine which classifier is more accurate (based on confidence)
                if nn_confidence > getattr(azure_confidence, neural_network_sentiment, 0):
                    print("\nThe Neural Network classifier appears more confident in its prediction.")
                else:
                    print("\nThe Azure Text Analytics classifier appears more confident in its prediction.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsing only Azure Text Analytics for sentiment analysis...")
        analyze_with_azure()


if __name__ == "__main__":
    main() 