import numpy as np
from math import exp
from math import log2
from numpy.linalg import inv
 
class MyLinearUnivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0

    # learn a linear univariate regression model by using training inputs (x) and outputs (y) 
    def fit(self, x, y):
        sx = sum(x)
        sy = sum(y)
        sx2 = sum(i * i for i in x)
        sxy = sum(i * j for (i,j) in zip(x, y))

        # The formulas derived from minimizing the sum of squared errors (SSE) are:
        # w1 = (n * Σ(xy) - Σx * Σy) / (n * Σ(x²) - (Σx)²)
        w1 = (len(x) * sxy - sx * sy) / (len(x) * sx2 - sx * sx)

        # w0 = (Σy - w1 * Σx) / n
        w0 = (sy - w1 * sx) / len(x)
        self.intercept_, self.coef_ =  w0, w1

    # predict the outputs for some new inputs (by using the learnt model)
    def predict(self, x):
        if (isinstance(x[0], list)):
            return [self.intercept_ + self.coef_ * val[0] for val in x]
        else:
            return [self.intercept_ + self.coef_ * val for val in x]
        

class MyLinearBivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0

    # learn a linear bivariate regression model by using training inputs (x) and outputs (y) 
    def fit(self, x, y):
        # For bivariate regression using least squares method:
        # We want to minimize E = Σ(yi - (w0 + w1*x1i + w2*x2i))²
        # Taking partial derivatives and setting them to 0:
        # ∂E/∂w0 = -2Σ(yi - (w0 + w1*x1i + w2*x2i)) = 0
        # ∂E/∂w1 = -2Σ(x1i(yi - (w0 + w1*x1i + w2*x2i))) = 0
        # ∂E/∂w2 = -2Σ(x2i(yi - (w0 + w1*x1i + w2*x2i))) = 0
        
        # Extract the two features
        x1 = [val[0] for val in x]
        x2 = [val[1] for val in x]
        
        # Number of samples
        n = len(x)
        
        # Calculate sums needed for the least squares method
        sum_x1 = sum(x1)
        sum_x2 = sum(x2)
        sum_y = sum(y)
        sum_x1_squared = sum(x1i * x1i for x1i in x1)
        sum_x2_squared = sum(x2i * x2i for x2i in x2)
        sum_x1y = sum(x1i * yi for x1i, yi in zip(x1, y))
        sum_x2y = sum(x2i * yi for x2i, yi in zip(x2, y))
        sum_x1x2 = sum(x1i * x2i for x1i, x2i in zip(x1, x2))
        
        # From the partial derivatives, we get a system of 3 equations:
        # n*w0 + w1*Σx1 + w2*Σx2 = Σy
        # w0*Σx1 + w1*Σ(x1²) + w2*Σ(x1x2) = Σ(x1y)
        # w0*Σx2 + w1*Σ(x1x2) + w2*Σ(x2²) = Σ(x2y)
        
        # Solve for w1 and w2 first using substitution method
        # This gives us:
        denominator = (n * sum_x1_squared - sum_x1 * sum_x1) * (n * sum_x2_squared - sum_x2 * sum_x2) - \
                     (n * sum_x1x2 - sum_x1 * sum_x2) * (n * sum_x1x2 - sum_x1 * sum_x2)
        
        # Check for numerical stability
        epsilon = 1e-10
        if abs(denominator) < epsilon:
            # If system is ill-conditioned, use individual least squares for each variable
            w1 = (n * sum_x1y - sum_x1 * sum_y) / (n * sum_x1_squared - sum_x1 * sum_x1) if abs(n * sum_x1_squared - sum_x1 * sum_x1) > epsilon else 0
            w2 = (n * sum_x2y - sum_x2 * sum_y) / (n * sum_x2_squared - sum_x2 * sum_x2) if abs(n * sum_x2_squared - sum_x2 * sum_x2) > epsilon else 0
        else:
            # Calculate w1 and w2 using the least squares formulas
            w1 = ((n * sum_x1y - sum_x1 * sum_y) * (n * sum_x2_squared - sum_x2 * sum_x2) - \
                  (n * sum_x2y - sum_x2 * sum_y) * (n * sum_x1x2 - sum_x1 * sum_x2)) / denominator
            
            w2 = ((n * sum_x2y - sum_x2 * sum_y) * (n * sum_x1_squared - sum_x1 * sum_x1) - \
                  (n * sum_x1y - sum_x1 * sum_y) * (n * sum_x1x2 - sum_x1 * sum_x2)) / denominator
        
        # Finally calculate w0 using the first equation
        w0 = (sum_y - w1 * sum_x1 - w2 * sum_x2) / n
        
        self.intercept_ = w0
        self.coef_ = np.array([w1, w2])

    # predict the outputs for some new inputs (by using the learnt model)
    def predict(self, x):
        if isinstance(x[0], list) or isinstance(x[0], np.ndarray):
            return [self.intercept_ + self.coef_[0] * val[0] + self.coef_[1] * val[1] for val in x]
        return [self.intercept_ + self.coef_[0] * x[0] + self.coef_[1] * x[1]]
        
        

