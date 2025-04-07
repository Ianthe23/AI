import numpy as np
import matplotlib.pyplot as plt
import os

"""
Custom Logistic Regression Implementation for Breast Cancer Classification
This program classifies breast cancer tissue samples as malignant or benign
based on radius and texture features from mammography data.
"""

class CustomLogisticRegression:
    """
    Custom implementation of logistic regression classifier
    """
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias"""
        self.weights = [0.0] * n_features
        self.bias = 0
        
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        Parameters:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.initialize_parameters(n_features)
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Linear combination: z = X·w + b
            z = []
            for i in range(n_samples):
                z_i = self.bias
                for j in range(n_features):
                    z_i += self.weights[j] * X[i][j]
                z.append(z_i)
            
            # Predictions using sigmoid
            y_pred = [self.sigmoid(z_i) for z_i in z]
            
            # Calculate gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += X[i][j] * error
                db += error
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= (self.learning_rate * dw[j]) / n_samples
            self.bias -= (self.learning_rate * db) / n_samples
            
        return self
            
    def predict_proba(self, X):
        """Predict probability of class 1"""
        n_samples, n_features = X.shape
        z = []
        for i in range(n_samples):
            z_i = self.bias
            for j in range(n_features):
                z_i += self.weights[j] * X[i][j]
            z.append(z_i)
        return [self.sigmoid(z_i) for z_i in z]
    
    def predict(self, X, threshold=0.5):
        """Predict class labels (0 or 1)"""
        probabilities = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probabilities]


class CustomStandardScaler:
    """
    Custom implementation of StandardScaler
    Standardizes features by removing the mean and scaling to unit variance
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """Compute mean and standard deviation of features"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
        
    def transform(self, X):
        """Standardize features"""
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit to data, then transform it"""
        self.fit(X)
        return self.transform(X)


# Load the Wisconsin Breast Cancer dataset
data_path = os.path.join('..', 'data', 'breast-cancer-diagnostic', 'wdbc.data')
# Alternative paths to try if the above doesn't work
alt_path1 = 'C:/Users/Ivona/Facultate/AI/lab6/data/breast-cancer-diagnostic/wdbc.data'
alt_path2 = '../data/breast-cancer-diagnostic/wdbc.data'

# Try different paths
if os.path.exists(data_path):
    file_path = data_path
elif os.path.exists(alt_path1):
    file_path = alt_path1
elif os.path.exists(alt_path2):
    file_path = alt_path2
else:
    # If none of the paths work, raise an error
    raise FileNotFoundError(f"Could not find the dataset file. Please update the path in the code. Tried: {data_path}, {alt_path1}, {alt_path2}")

print(f"Loading data from: {file_path}")

# Load and process the data
data = []
with open(file_path, 'r') as f:
    for line in f:
        values = line.strip().split(',')
        data.append(values)

# Extract features (radius, texture) and labels (M/B)
features = np.array([[float(row[2]), float(row[3])] for row in data])  # Radius, Texture
labels = np.array([1 if row[1] == 'M' else 0 for row in data])  # 1 = Malignant, 0 = Benign

# Normalize features using custom scaler
scaler = CustomStandardScaler()
features_scaled = scaler.fit_transform(features)

# Train the custom logistic regression model
model = CustomLogisticRegression(learning_rate=0.1, max_iterations=1000)
model.fit(features_scaled, labels)

# Display model information
print("Custom Logistic Regression Model for Breast Cancer Classification")
print("--------------------------------------------------------------")
print(f"Number of samples: {len(features)}")
print(f"Number of malignant samples: {np.sum(labels == 1)}")
print(f"Number of benign samples: {np.sum(labels == 0)}")

print("\nModel Parameters:")
print(f"Intercept (bias): {model.bias:.4f}")
print(f"Coefficient for radius: {model.weights[0]:.4f}")
print(f"Coefficient for texture: {model.weights[1]:.4f}")

# Evaluate the model on training data
y_pred = np.array(model.predict(features_scaled))  # Convert to numpy array
accuracy = np.mean(y_pred == labels)
print(f"\nTraining Accuracy: {accuracy:.4f}")

# Calculate confusion matrix manually
TP = np.sum((y_pred == 1) & (labels == 1))
TN = np.sum((y_pred == 0) & (labels == 0))
FP = np.sum((y_pred == 1) & (labels == 0))
FN = np.sum((y_pred == 0) & (labels == 1))

print("\nConfusion Matrix:")
print(f"[[{TN}, {FP}],")
print(f" [{FN}, {TP}]]")

# Calculate metrics
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\nClassification Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Create visualization of data points and decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(features[labels==0][:, 0], features[labels==0][:, 1], color='green', 
            marker='o', label='Benign', alpha=0.6)
plt.scatter(features[labels==1][:, 0], features[labels==1][:, 1], color='red', 
            marker='x', label='Malignant', alpha=0.6)
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Breast Cancer Classification - Custom Logistic Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Create a mesh grid to plot decision boundary
x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Scale the grid points
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler.transform(grid_points)

# Get predictions for each point on the grid
Z = model.predict(grid_points_scaled)
Z = np.array(Z)  # Convert list to numpy array
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contour(xx, yy, Z, colors='black', linestyles=['-'], levels=[0.5])

# Mark the specific test point (radius=18, texture=10)
test_point = np.array([[18, 10]])
plt.scatter(test_point[0, 0], test_point[0, 1], color='blue', marker='*', 
            s=200, label='Test Point (18, 10)')
plt.legend()
plt.savefig('pb2/breast_cancer_custom_classification.png')
plt.show()

# Function to classify a new sample using the custom model
def predict_sample(radius, texture):
    # Scale the input using the custom scaler
    sample = np.array([[radius, texture]])
    sample_scaled = scaler.transform(sample)
    
    # Get prediction and probability
    probability = model.predict_proba(sample_scaled)[0]
    prediction = 1 if probability >= 0.5 else 0
    
    result = "Malignant" if prediction == 1 else "Benign"
    return result, probability

# Classify the specific sample (radius=18, texture=10)
sample_radius = 18
sample_texture = 10
result, probability = predict_sample(sample_radius, sample_texture)

print(f"\nClassification of sample with radius={sample_radius}, texture={sample_texture}:")
print(f"Prediction: {result}")
print(f"Probability of being malignant: {probability:.4f}")
print(f"Probability of being benign: {1-probability:.4f}")

# Compute equation for the decision boundary
w1, w2 = model.weights
b = model.bias
print("\nDecision Boundary Equation:")
print(f"w1*x1 + w2*x2 + b = 0")
print(f"{w1:.4f}*radius_scaled + {w2:.4f}*texture_scaled + {b:.4f} = 0")

# Explain the prediction
print("\nExplanation:")
if result == "Malignant":
    print("The sample is classified as MALIGNANT because:")
    print(f"- The radius value of {sample_radius} is relatively high")
    if sample_texture > np.mean(features[:, 1]):
        print(f"- The texture value of {sample_texture} is also elevated")
    print(f"- The model is {probability:.1%} confident in this classification")
else:
    print("The sample is classified as BENIGN because:")
    if sample_radius < np.mean(features[:, 0]):
        print(f"- The radius value of {sample_radius} is relatively low")
    if sample_texture < np.mean(features[:, 1]):
        print(f"- The texture value of {sample_texture} is relatively low")
    print(f"- The model is {1-probability:.1%} confident in this classification") 