import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

"""
Breast Cancer Classification using Logistic Regression
This program classifies breast cancer tissue samples as malignant or benign
based on radius and texture values from mammography data.
"""

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

data = []
with open(file_path, 'r') as f:
    for line in f:
        values = line.strip().split(',')
        data.append(values)

# Extract features (radius, texture) and labels (M/B)
features = np.array([[float(row[2]), float(row[3])] for row in data])  # Radius, Texture
labels = np.array([1 if row[1] == 'M' else 0 for row in data])  # 1 = Malignant, 0 = Benign

# Normalize/standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train the logistic regression model on the full dataset
# (for a real application, you would use train/test split)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(features_scaled, labels)

# Display model information
print("Logistic Regression Model for Breast Cancer Classification")
print("--------------------------------------------------------")
print(f"Number of samples: {len(features)}")
print(f"Number of malignant samples: {np.sum(labels == 1)}")
print(f"Number of benign samples: {np.sum(labels == 0)}")

print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient for radius: {model.coef_[0][0]:.4f}")
print(f"Coefficient for texture: {model.coef_[0][1]:.4f}")

# Create visualization of data points and decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(features[labels==0][:, 0], features[labels==0][:, 1], color='green', 
            marker='o', label='Benign', alpha=0.6)
plt.scatter(features[labels==1][:, 0], features[labels==1][:, 1], color='red', 
            marker='x', label='Malignant', alpha=0.6)
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Breast Cancer Classification by Radius and Texture')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot decision boundary
x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Scale the grid points
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler.transform(grid_points)

# Get predictions for each point on the grid
Z = model.predict(grid_points_scaled).reshape(xx.shape)

# Plot decision boundary
plt.contour(xx, yy, Z, colors='black', linestyles=['-'], levels=[0.5])

# Mark the specific test point (radius=18, texture=10)
test_point = np.array([[18, 10]])
plt.scatter(test_point[0, 0], test_point[0, 1], color='blue', marker='*', 
            s=200, label='Test Point (18, 10)')
plt.legend()
plt.savefig('pb2/breast_cancer_simple_classification.png')
plt.show()

# Function to classify a new sample
def predict_sample(radius, texture):
    # Scale the input using the same scaler
    sample = np.array([[radius, texture]])
    sample_scaled = scaler.transform(sample)
    
    # Get prediction and probability
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]  # Probability of malignant
    
    return "Malignant" if prediction == 1 else "Benign", probability

# Classify the specific sample (radius=18, texture=10)
sample_radius = 18
sample_texture = 10
result, probability = predict_sample(sample_radius, sample_texture)

print("\nClassification of sample with radius=18, texture=10:")
print(f"Prediction: {result}")
print(f"Probability of being malignant: {probability:.4f}")
print(f"Probability of being benign: {1-probability:.4f}")

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