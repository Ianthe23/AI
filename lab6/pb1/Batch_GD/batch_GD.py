import numpy as np

# Custom implementation of Batch Gradient Descent
class MyBatchGDRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, batch_size=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size  # If None, use full batch
        self.coef_ = None
        self.intercept_ = None
        self.errors_ = []  # Store errors for each batch
    
    def fit(self, X, y):
        # Initialize weights
        n_samples, n_features = len(X), len(X[0]) if isinstance(X[0], list) else 1
        
        if n_features == 1 and not isinstance(X[0], list):
            X = [[x] for x in X]
        
        self.coef_ = [0.0] * n_features
        self.intercept_ = 0
        self.errors_ = []
        
        # Set batch size to full dataset if not specified
        if self.batch_size is None or self.batch_size > n_samples:
            self.batch_size = n_samples
        
        # Gradient Descent
        for _ in range(self.iterations):
            # Create random batches
            indices = np.random.permutation(n_samples)
            epoch_error = 0.0
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = [X[i] for i in batch_indices]
                y_batch = [y[i] for i in batch_indices]
                
                # Compute predictions for batch
                y_pred = []
                for x in X_batch:
                    pred = self.intercept_
                    for j in range(n_features):
                        pred += self.coef_[j] * x[j]
                    y_pred.append(pred)
                
                # Calculate gradients for batch
                dw = [0.0] * n_features
                db = 0
                batch_error = 0.0
                
                for i, x in enumerate(X_batch):
                    error = y_pred[i] - y_batch[i]
                    batch_error += error * error  # Sum of squared errors

                for j in range(n_features):
                    dw[j] += batch_error/len(X_batch) * x[j]
                db += batch_error/len(X_batch)

                
                # Update weights using batch gradients
                for j in range(n_features):
                    self.coef_[j] -= (self.learning_rate * dw[j]) / len(X_batch)
                self.intercept_ -= (self.learning_rate * db) / len(X_batch)
                
                # Store batch error
                self.errors_.append(batch_error / len(X_batch))
                epoch_error += batch_error
            
            # Store average error for this epoch
            self.errors_.append(epoch_error / n_samples)
        
        return self
    
    def partial_fit(self, X, y):
        n_samples, n_features = len(X), len(X[0]) if isinstance(X[0], list) else 1
        
        if n_features == 1 and not isinstance(X[0], list):
            X = [[x] for x in X]
        
        self.coef_ = [0.0] * n_features
        self.intercept_ = 0
        self.errors_ = []

        # Gradient Descent
        for _ in range(self.iterations):
            # Create random batches
            indices = np.random.permutation(n_samples)
            epoch_error = 0.0
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = [X[i] for i in batch_indices]
                y_batch = [y[i] for i in batch_indices]

                # Compute predictions for batch
                y_pred = []
                for x in X_batch:
                    pred = self.intercept_
                    for j in range(n_features):
                        pred += self.coef_[j] * x[j]
                    y_pred.append(pred)
                
                # Calculate gradients for batch
                dw = [0.0] * n_features
                db = 0
                batch_error = 0.0
                
                for i, x in enumerate(X_batch):
                    error = y_pred[i] - y_batch[i]
                    batch_error += error * error  # Sum of squared errors
                    for j in range(n_features):
                        dw[j] += error * x[j]
                    db += error
                
                # Update weights using batch gradients
                for j in range(n_features):
                    self.coef_[j] -= (self.learning_rate * dw[j]) / len(X_batch)
                self.intercept_ -= (self.learning_rate * db) / len(X_batch)
                
                # Store batch error
                self.errors_.append(batch_error / len(X_batch))
                epoch_error += batch_error
            
            # Store average error for this epoch
            self.errors_.append(epoch_error / n_samples)
                
        
    def predict(self, X):
        if not isinstance(X[0], list):
            X = [[x] for x in X]
        
        predictions = []
        for x in X:
            pred = self.intercept_
            for j in range(len(self.coef_)):
                pred += self.coef_[j] * x[j]
            predictions.append(pred)
        return predictions