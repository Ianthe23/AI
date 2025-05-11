import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Custom implementations of restricted functions
def custom_exp(x):
    # Custom exponential implementation using Taylor series
    result = np.ones_like(x)
    term = np.ones_like(x)
    for i in range(1, 20):  # Using 20 terms for reasonable accuracy
        term = term * x / i
        result += term
    return result

def custom_sum(x, axis=None, keepdims=False):
    """
    Custom sum implementation that mimics numpy.sum
    
    Args:
        x: Input array
        axis: Axis along which to sum (None for all elements)
        keepdims: Whether to keep the dimensions of the input array
    """
    if axis is None:
        result = sum(x.flatten())
        if keepdims:
            return np.array([[result]])
        return result
    
    # Handle axis-specific sums
    if axis == 0:
        result = np.array([sum(x[:, i]) for i in range(x.shape[1])])
    elif axis == 1:
        result = np.array([sum(x[i, :]) for i in range(x.shape[0])])
    else:
        result = sum(x.flatten())
    
    # Reshape result if keepdims is True
    if keepdims:
        if axis == 0:
            result = result.reshape(1, -1)
        elif axis == 1:
            result = result.reshape(-1, 1)
    
    return result

def custom_dot(a, b):
    # Custom matrix multiplication implementation
    if len(a.shape) == 1 and len(b.shape) == 1:
        return sum(a[i] * b[i] for i in range(len(a)))
    
    result = np.zeros((a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            result[i, j] = sum(a[i, k] * b[k, j] for k in range(a.shape[1]))
    return result

def custom_log(x):
    # Custom natural logarithm implementation using Taylor series
    # Note: This is a simplified version and works best for x close to 1
    x = np.clip(x, 1e-10, None)  # Avoid log(0)
    y = (x - 1) / (x + 1)
    result = 0
    for i in range(1, 20, 2):  # Using 20 terms
        result += (y ** i) / i
    return 2 * result

# The message we need to classify
MESSAGE_TO_CLASSIFY = "By choosing a bike over a car, I'm reducing my environmental footprint. Cycling promotes eco-friendly transportation, and I'm proud to be part of that movement."

# Text preprocessing function (same as in main script)
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


# Implementation of a neural network from scratch
class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
        self.loss_history = []
        self.accuracy_history = []
        
    def sigmoid(self, x):
        # Sigmoid activation function using custom_exp
        return 1 / (1 + custom_exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def softmax(self, x):
        # Softmax activation using custom_exp
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = custom_exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Forward pass using custom_dot
        self.z1 = custom_dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = custom_dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        # Compute cross-entropy loss using custom_log
        m = y_true.shape[0]
        log_likelihood = -custom_log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = custom_sum(log_likelihood) / m
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)
    
    def backward(self, X, y):
        """
        Backpropagation algorithm to compute gradients and update network parameters.
        
        Args:
            X: Input features (shape: batch_size x input_size)
            y: True labels in one-hot format (shape: batch_size x output_size)
        """
        # Get batch size for normalization
        m = X.shape[0]
        
        # Step 1: Compute output layer error (delta2)
        # This represents how much each output neuron contributed to the error
        delta2 = self.a2.copy()  # Start with the output layer activations
        # For each sample, subtract 1 from the predicted class (where true label is 1)
        delta2[range(m), y.argmax(axis=1)] -= 1
        # Normalize by batch size
        delta2 /= m
        
        # Step 2: Compute gradients for output layer weights and biases
        # dW2 = (a1^T * delta2) - gradient of loss with respect to W2
        # This shows how much each hidden neuron contributed to the output error
        dW2 = custom_dot(self.a1.T, delta2)
        # db2 = sum(delta2) - gradient of loss with respect to b2
        # This shows how much each bias contributed to the output error
        db2 = custom_sum(delta2, axis=0, keepdims=True)
        
        # Step 3: Compute hidden layer error (delta1)
        # This propagates the error back to the hidden layer
        # delta1 = (delta2 * W2^T) * sigmoid_derivative(a1)
        # First part: (delta2 * W2^T) - error propagated back through weights
        # Second part: * sigmoid_derivative(a1) - chain rule for sigmoid activation
        delta1 = custom_dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        
        # Step 4: Compute gradients for hidden layer weights and biases
        # dW1 = (X^T * delta1) - gradient of loss with respect to W1
        # This shows how much each input feature contributed to the hidden layer error
        dW1 = custom_dot(X.T, delta1)
        # db1 = sum(delta1) - gradient of loss with respect to b1
        # This shows how much each hidden layer bias contributed to the error
        db1 = custom_sum(delta1, axis=0, keepdims=True)
        
        # Step 5: Update network parameters using gradient descent
        # Move in the opposite direction of the gradient, scaled by learning rate
        self.W1 -= self.learning_rate * dW1  # Update input-to-hidden weights
        self.b1 -= self.learning_rate * db1  # Update hidden layer biases
        self.W2 -= self.learning_rate * dW2  # Update hidden-to-output weights
        self.b2 -= self.learning_rate * db2  # Update output layer biases
    
    def train(self, X, y, epochs=100, batch_size=32, validation_data=None):
        num_samples = X.shape[0]
        num_batches = num_samples // batch_size
        
        # One-hot encode the target labels if they're not already
        if len(y.shape) == 1:
            encoder = OneHotEncoder(sparse_output=False)
            y = encoder.fit_transform(y.reshape(-1, 1))
        
        print(f"Training custom neural network for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            # Mini-batch training
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Calculate loss and accuracy
                batch_loss = self.compute_loss(y_batch, y_pred)
                batch_accuracy = self.compute_accuracy(y_batch, y_pred)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                
                # Backward pass
                self.backward(X_batch, y_batch)
            
            # Average loss and accuracy for the epoch
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            
            # Save history
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_accuracy = self.compute_accuracy(y_val, val_pred)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")
    
    def predict(self, X):
        # Make predictions
        return self.forward(X)
    
    def plot_history(self, save_path=None):
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.accuracy_history)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_history)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    print("Facebook Sentiment Analysis with Custom Neural Network")
    print("====================================================")
    
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
        
        # One-hot encode the labels
        encoder = OneHotEncoder(sparse_output=False)
        y_one_hot = encoder.fit_transform(df['encoded_sentiment'].values.reshape(-1, 1))
        
        # Display the label mapping
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        print("\nSentiment label mapping:")
        for idx, label in label_mapping.items():
            print(f"  {idx}: {label}")
        
        # Extract features with TF-IDF
        print("\nExtracting TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=1000)  # Limiting features for computational efficiency
        X_features = vectorizer.fit_transform(df['processed_text']).toarray()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, 
            y_one_hot,
            test_size=0.2,
            random_state=42
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Set up neural network architecture
        input_size = X_train.shape[1]
        hidden_size = 64
        output_size = y_train.shape[1]
        
        # Create and train custom neural network
        custom_nn = CustomNeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            learning_rate=0.01
        )
        
        # Train the model with validation data
        custom_nn.train(
            X_train, 
            y_train, 
            epochs=50,  # Fewer epochs for demonstration
            batch_size=32,
            validation_data=(X_test, y_test)
        )
        
        # Plot and save training history
        custom_nn.plot_history(save_path="custom_nn_history.png")
        
        # Evaluate on test set
        y_pred = custom_nn.predict(X_test)
        test_loss = custom_nn.compute_loss(y_test, y_pred)
        test_accuracy = custom_nn.compute_accuracy(y_test, y_pred)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes, 
                                 target_names=label_encoder.classes_,
                                 labels=range(len(label_encoder.classes_))))
        
        # Analyze the given message
        print(f"\n=== Analyzing message with Custom Neural Network ===")
        print(f"Message: \"{MESSAGE_TO_CLASSIFY}\"")
        
        # Preprocess and vectorize the message
        processed_message = preprocess_text(MESSAGE_TO_CLASSIFY)
        X_message = vectorizer.transform([processed_message]).toarray()
        
        # Predict sentiment
        message_pred = custom_nn.predict(X_message)
        predicted_class_idx = np.argmax(message_pred, axis=1)[0]
        predicted_sentiment = label_mapping.get(predicted_class_idx, f"unknown ({predicted_class_idx})")
        confidence = message_pred[0][predicted_class_idx]
        
        print(f"Predicted sentiment: {predicted_sentiment}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 