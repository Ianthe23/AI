import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Helper function to load the breast cancer data
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    
    # Extract the features we need (ID, diagnosis, mean radius, mean texture)
    extracted_data = []
    for row in data:
        # Column 0: ID, Column 1: Diagnosis (M/B), Column 2: mean radius, Column 3: mean texture
        extracted_data.append([row[0], row[1], float(row[2]), float(row[3])])
    
    return extracted_data

# Locate the data file
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

# Load the data
data = load_data(file_path)

# Extract features and labels
X = np.array([[row[2], row[3]] for row in data])  # Radius and texture
y = np.array([1 if row[1] == 'M' else 0 for row in data])  # 1 for Malignant, 0 for Benign

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80/20)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices, test_indices = indices[:train_size], indices[train_size:]

X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient for radius: {model.coef_[0][0]:.4f}")
print(f"Coefficient for texture: {model.coef_[0][1]:.4f}")
print(f"\nLogistic Regression Equation: P(malignant) = 1 / (1 + exp(-({model.intercept_[0]:.4f} + {model.coef_[0][0]:.4f} * radius_scaled + {model.coef_[0][1]:.4f} * texture_scaled)))")

# Visualize the data and decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Benign', alpha=0.6, marker='o', color='green')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='Malignant', alpha=0.6, marker='x', color='red')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Breast Cancer Diagnosis based on Radius and Texture')
plt.legend()
plt.grid(True, alpha=0.3)

# Create a mesh grid to plot decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Scale the mesh grid points
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)

# Predict using the model
Z = model.predict(mesh_points_scaled)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contour(xx, yy, Z, colors='k', linestyles=['-'], levels=[0.5])

# Mark the test point
test_point = np.array([[18, 10]])
plt.scatter(test_point[0, 0], test_point[0, 1], color='blue', marker='*', 
            s=200, label='Test Point (18, 10)')
plt.legend()
plt.savefig('pb2/breast_cancer_classifier.png')
plt.show()

# Make a prediction for a new tissue sample
def predict_new_sample(radius, texture):
    # Scale the input features
    sample = np.array([[radius, texture]])
    sample_scaled = scaler.transform(sample)
    
    # Make prediction
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]
    
    result = "Malignant" if prediction == 1 else "Benign"
    return result, probability

# Predict a sample with texture = 10 and radius = 18
sample_radius = 18
sample_texture = 10
result, probability = predict_new_sample(sample_radius, sample_texture)

print(f"\nPrediction for new tissue sample (radius = {sample_radius}, texture = {sample_texture}):")
print(f"Classification: {result}")
print(f"Probability of being malignant: {probability:.4f}")
print(f"Probability of being benign: {1-probability:.4f}") 